"""维度下钻经营归属：分区明细批次与差异报告的领域服务。

补齐维度下钻"按时段和父层查询经营归属数据"的能力，整条链路如下：

1. 经营区域归属按生效区间维护，子类（省份）只取当时有效的经营区域；
2. 打开某时段+父层的下钻明细时生成「分区明细批次」：按父层筛选子地区后，
   把分区与原始成交行关联起来。批次是不可变快照，原始成交行本身不被改写；
3. 迟到更正在当前有效批次上把成交行改派到其他子地区，生成新版本批次，
   旧批次标记「已更正」，更正先后由单调序号可查；
4. 差异报告从同一批次版本逐省重算实际合计与达到核查门槛的名单，并校验
   分区合计，保证分区和金额来自同一版本，不会混用更正前后的归属；
5. 报告确认出数幂等（同一报告只能出数一次），确认时校验批次版本仍是
   当前有效版本；已确认的历史报告可按其锁定的批次版本复算。

所有方法都是确定性的，可直接用于测试；HTTP 层只做参数透传和错误映射。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 领域错误
# ============================================================


class DrilldownError(Exception):
    """维度下钻经营归属的领域错误基类。"""


class BatchNotFoundError(DrilldownError):
    """分区明细批次不存在。"""


class PartitionNotFoundError(DrilldownError):
    """批次内没有该子地区的分区。"""


class ReportNotFoundError(DrilldownError):
    """差异报告不存在。"""


class StaleBatchError(DrilldownError):
    """批次版本已被更正取代，不能基于旧版本继续更正或确认出数。"""


class AlreadyConfirmedError(DrilldownError):
    """报告已确认出数，同一报告只能出数一次。"""


class CorrectionError(DrilldownError):
    """迟到更正内容非法。"""


class ReportStateError(DrilldownError):
    """报告状态不允许当前操作。"""


# ============================================================
# 领域模型
# ============================================================


@dataclass(frozen=True)
class RegionAssignment:
    """经营区域归属：省份在 [effective_from, effective_to) 时段内隶属父层。

    时段采用可字典序比较的零填充字符串（如 2026-09、2026-10）。
    """

    province: str
    parent_region: str
    effective_from: str
    effective_to: str | None = None

    def covers(self, period: str) -> bool:
        if period < self.effective_from:
            return False
        return self.effective_to is None or period < self.effective_to


@dataclass(frozen=True)
class TransactionRow:
    """原始成交行：归属以写入时为准，后续更正不改写原始行。"""

    txn_id: str
    period: str
    province: str
    amount: float


@dataclass(frozen=True)
class PartitionDetail:
    """分区明细：一个子地区在批次内的合计金额与其成交行关联。"""

    province: str
    amount: float
    txn_ids: tuple[str, ...]


@dataclass
class PartitionBatch:
    """分区明细批次：某时段+父层的一次性不可变快照，更正产生新版本。

    partitions 构建后不再变化；只有 status 会随被更正而从「有效」翻为「已更正」。
    """

    batch_id: str
    period: str
    parent_region: str
    version: int
    partitions: tuple[PartitionDetail, ...]
    created_seq: int
    supersedes: str | None = None
    status: str = "有效"  # 有效 / 已更正

    @property
    def total_amount(self) -> float:
        return round(sum(part.amount for part in self.partitions), 2)


@dataclass(frozen=True)
class ProvinceMetric:
    """差异报告中的单省指标：实际合计、所用成交行、是否达到核查门槛。"""

    province: str
    amount: float
    txn_ids: tuple[str, ...]
    reaches_threshold: bool


@dataclass
class DiffReport:
    """差异报告：从同一批次版本逐省计算实际合计与核查名单。

    报告锁定 batch_id + batch_version，确认后即为历史报告，可复算核对。
    """

    report_id: str
    batch_id: str
    batch_version: int
    period: str
    parent_region: str
    threshold: float
    province_metrics: tuple[ProvinceMetric, ...]
    audit_list: tuple[str, ...]
    total_amount: float
    audit_total: float
    validated: bool
    validation_errors: tuple[str, ...]
    created_seq: int
    status: str = "待确认"  # 待确认 / 已确认
    confirmed_seq: int | None = None


# ============================================================
# 领域服务
# ============================================================


class DrilldownAttributionService:
    """维度下钻经营归属服务：批次、更正、差异报告、确认与复算。"""

    def __init__(
        self,
        assignments: list[RegionAssignment],
        transactions: list[TransactionRow],
    ) -> None:
        self._assignments = list(assignments)
        self._transactions = {txn.txn_id: txn for txn in transactions}
        self._batches: dict[str, PartitionBatch] = {}
        self._current: dict[tuple[str, str], str] = {}  # (时段, 父层) -> 当前有效批次
        self._reports: dict[str, DiffReport] = {}
        self._corrections: list[dict[str, Any]] = []
        self._seq = 0
        self._batch_seq = 0
        self._report_seq = 0
        self._lock = threading.RLock()

    # ---- 内部工具 ----

    def _tick(self) -> int:
        """单调序号：批次、更正、报告、确认的先后顺序都靠它核对。"""
        self._seq += 1
        return self._seq

    def _get_batch(self, batch_id: str) -> PartitionBatch:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise BatchNotFoundError(f"分区明细批次 {batch_id} 不存在")
        return batch

    def _get_report(self, report_id: str) -> DiffReport:
        report = self._reports.get(report_id)
        if report is None:
            raise ReportNotFoundError(f"差异报告 {report_id} 不存在")
        return report

    def _sum_txns(self, txn_ids: tuple[str, ...] | list[str]) -> float:
        return round(sum(self._transactions[tid].amount for tid in txn_ids), 2)

    # ---- 经营区域归属（当时有效） ----

    def effective_children(self, period: str, parent_region: str) -> list[str]:
        """某时段下父层当时有效的子地区。"""
        children = {
            item.province
            for item in self._assignments
            if item.parent_region == parent_region and item.covers(period)
        }
        return sorted(children)

    def region_tree(self, period: str) -> dict[str, list[str]]:
        """某时段全部父层 -> 当时有效子地区。"""
        tree: dict[str, set[str]] = {}
        for item in self._assignments:
            if item.covers(period):
                tree.setdefault(item.parent_region, set()).add(item.province)
        return {parent: sorted(tree[parent]) for parent in sorted(tree)}

    # ---- 分区明细批次 ----

    def open_batch(self, period: str, parent_region: str) -> PartitionBatch:
        """打开下钻明细：返回该时段+父层当前有效批次，没有则按当时有效归属生成 v1。

        按父层筛选子地区之后，把分区与原始成交行关联起来；重复打开返回同一批次。
        """
        with self._lock:
            key = (period, parent_region)
            current_id = self._current.get(key)
            if current_id is not None:
                return self._batches[current_id]
            children = set(self.effective_children(period, parent_region))
            by_province: dict[str, list[str]] = {}
            for txn in self._transactions.values():
                if txn.period == period and txn.province in children:
                    by_province.setdefault(txn.province, []).append(txn.txn_id)
            return self._new_batch(period, parent_region, version=1,
                                   supersedes=None, by_province=by_province)

    def _new_batch(
        self,
        period: str,
        parent_region: str,
        *,
        version: int,
        supersedes: str | None,
        by_province: dict[str, list[str]],
    ) -> PartitionBatch:
        partitions = []
        for province in sorted(by_province):
            txn_ids = tuple(sorted(by_province[province]))
            if not txn_ids:
                continue
            partitions.append(PartitionDetail(
                province=province,
                amount=self._sum_txns(txn_ids),
                txn_ids=txn_ids,
            ))
        self._batch_seq += 1
        batch = PartitionBatch(
            batch_id=f"PB-{self._batch_seq:04d}",
            period=period,
            parent_region=parent_region,
            version=version,
            partitions=tuple(partitions),
            created_seq=self._tick(),
            supersedes=supersedes,
        )
        self._batches[batch.batch_id] = batch
        self._current[(period, parent_region)] = batch.batch_id
        return batch

    def get_batch(self, batch_id: str) -> PartitionBatch:
        return self._get_batch(batch_id)

    def current_batch(self, period: str, parent_region: str) -> PartitionBatch | None:
        current_id = self._current.get((period, parent_region))
        return self._batches[current_id] if current_id else None

    def province_transactions(
        self,
        batch_id: str,
        province: str,
        *,
        page: int = 1,
        size: int = 20,
    ) -> dict[str, Any]:
        """省份页：批次内某子地区合计所用的原始成交行（分页）。

        每行同时给出原始归属省份和本批次版本下的归属省份，更正痕迹可查。
        """
        batch = self._get_batch(batch_id)
        part = next((p for p in batch.partitions if p.province == province), None)
        if part is None:
            raise PartitionNotFoundError(
                f"批次 {batch_id} 中没有子地区「{province}」的分区")
        page = max(page, 1)
        size = max(size, 1)
        start = (page - 1) * size
        ids = part.txn_ids[start:start + size]
        items = [
            {
                "txn_id": txn.txn_id,
                "period": txn.period,
                "province": txn.province,  # 原始成交行上的归属
                "assigned_province": province,  # 本批次版本下的归属
                "amount": txn.amount,
            }
            for txn in (self._transactions[tid] for tid in ids)
        ]
        return {
            "batch_id": batch.batch_id,
            "batch_version": batch.version,
            "period": batch.period,
            "parent_region": batch.parent_region,
            "province": province,
            "partition_amount": part.amount,
            "total_rows": len(part.txn_ids),
            "page": page,
            "size": size,
            "items": items,
        }

    # ---- 迟到更正 ----

    def apply_correction(
        self,
        batch_id: str,
        moves: list[dict[str, str]],
    ) -> PartitionBatch:
        """迟到更正：把成交行改派到同属父层的其他子地区，生成新版本批次。

        只能基于当前有效批次更正；目标省份必须是该时段父层当时有效的子地区；
        父层内改派不改变父层总额。原始成交行不改写，旧批次标记「已更正」。
        """
        with self._lock:
            base = self._get_batch(batch_id)
            key = (base.period, base.parent_region)
            if self._current.get(key) != batch_id:
                raise StaleBatchError(
                    f"批次 {batch_id} 已被更正，只能基于当前有效批次再次更正")
            if not moves:
                raise CorrectionError("迟到更正至少要包含一条成交行改派")

            by_province = {p.province: list(p.txn_ids) for p in base.partitions}
            effective = set(self.effective_children(base.period, base.parent_region))
            applied: list[dict[str, str]] = []
            for move in moves:
                txn_id = str(move.get("txn_id") or "")
                to_province = str(move.get("to_province") or "")
                from_province = next(
                    (prov for prov, ids in by_province.items() if txn_id in ids), None)
                if from_province is None:
                    raise CorrectionError(
                        f"成交行 {txn_id} 不在批次 {batch_id} 的分区中")
                if to_province not in effective:
                    raise CorrectionError(
                        f"子地区「{to_province}」在时段 {base.period} 不属于"
                        f"父层「{base.parent_region}」当时有效的经营区域")
                if to_province == from_province:
                    raise CorrectionError(
                        f"成交行 {txn_id} 已在子地区「{to_province}」中，无需改派")
                by_province[from_province].remove(txn_id)
                by_province.setdefault(to_province, []).append(txn_id)
                applied.append({
                    "txn_id": txn_id,
                    "from_province": from_province,
                    "to_province": to_province,
                    "amount": self._transactions[txn_id].amount,
                })

            batch = self._new_batch(
                base.period, base.parent_region,
                version=base.version + 1,
                supersedes=base.batch_id,
                by_province=by_province,
            )
            base.status = "已更正"
            self._corrections.append({
                "seq": batch.created_seq,
                "period": base.period,
                "parent_region": base.parent_region,
                "from_batch": base.batch_id,
                "to_batch": batch.batch_id,
                "moves": applied,
            })
            return batch

    def correction_log(self) -> list[dict[str, Any]]:
        """更正流水：按发生顺序记录，供核对更正先后。"""
        return [dict(item) for item in self._corrections]

    # ---- 差异报告 ----

    def compute_diff_report(self, batch_id: str, threshold: float) -> DiffReport:
        """用分区明细批次逐省计算差异报告。

        各省实际合计从批次关联的原始成交行重算（与省份页、父层批次同源），
        并校验与分区合计一致；达到核查门槛的省份进入核查名单。
        报告锁定批次版本，分区和金额必然来自同一版本。
        """
        if threshold <= 0:
            raise ReportStateError("核查门槛必须为正数")
        with self._lock:
            batch = self._get_batch(batch_id)
            metrics: list[ProvinceMetric] = []
            errors: list[str] = []
            for part in batch.partitions:
                missing = [tid for tid in part.txn_ids if tid not in self._transactions]
                if missing:
                    errors.append(
                        f"分区{part.province}关联的成交行缺失：{'、'.join(missing)}")
                    continue
                actual = self._sum_txns(part.txn_ids)
                if actual != part.amount:
                    errors.append(
                        f"分区{part.province}合计 {part.amount} 与成交行重算 {actual} 不一致")
                metrics.append(ProvinceMetric(
                    province=part.province,
                    amount=actual,
                    txn_ids=part.txn_ids,
                    reaches_threshold=actual >= threshold,
                ))
            audit_list = tuple(sorted(
                (m.province for m in metrics if m.reaches_threshold),
                key=lambda prov: (-next(m.amount for m in metrics if m.province == prov), prov),
            ))
            self._report_seq += 1
            report = DiffReport(
                report_id=f"DR-{self._report_seq:04d}",
                batch_id=batch.batch_id,
                batch_version=batch.version,
                period=batch.period,
                parent_region=batch.parent_region,
                threshold=threshold,
                province_metrics=tuple(metrics),
                audit_list=audit_list,
                total_amount=round(sum(m.amount for m in metrics), 2),
                audit_total=round(
                    sum(m.amount for m in metrics if m.reaches_threshold), 2),
                validated=not errors,
                validation_errors=tuple(errors),
                created_seq=self._tick(),
            )
            self._reports[report.report_id] = report
            return report

    def get_report(self, report_id: str) -> DiffReport:
        return self._get_report(report_id)

    def confirm_report(self, report_id: str) -> DiffReport:
        """确认出数：同一报告只能出数一次，且批次版本必须仍是当前有效版本。

        加锁保证并发确认只有第一个成功；若报告计算后归属被更正，则拒绝确认，
        避免混用更正前后的归属出数。
        """
        with self._lock:
            report = self._get_report(report_id)
            if report.status == "已确认":
                raise AlreadyConfirmedError(
                    f"差异报告 {report_id} 已确认出数，同一报告只能出数一次")
            current_id = self._current.get((report.period, report.parent_region))
            if current_id != report.batch_id:
                raise StaleBatchError(
                    f"差异报告 {report_id} 基于批次 {report.batch_id}"
                    f"(v{report.batch_version})，该时段+父层已更正为 {current_id}，"
                    f"不能混用更正前后的归属，请基于当前批次重算后再确认")
            report.status = "已确认"
            report.confirmed_seq = self._tick()
            return report

    def recalculate_report(self, report_id: str) -> dict[str, Any]:
        """复算历史报告：按报告锁定的批次版本重算，与已确认结果核对。"""
        with self._lock:
            report = self._get_report(report_id)
            if report.status != "已确认":
                raise ReportStateError(
                    f"差异报告 {report_id} 尚未确认出数，只有历史报告可以复算")
            batch = self._get_batch(report.batch_id)
            recomputed = {
                part.province: self._sum_txns(part.txn_ids)
                for part in batch.partitions
            }
            confirmed = {m.province: m.amount for m in report.province_metrics}
            recomputed_audit = sorted(
                (prov for prov, amount in recomputed.items()
                 if amount >= report.threshold),
                key=lambda prov: (-recomputed[prov], prov),
            )
            matches = recomputed == confirmed and recomputed_audit == list(report.audit_list)
            return {
                "report_id": report.report_id,
                "batch_id": batch.batch_id,
                "batch_version": batch.version,
                "batch_status": batch.status,
                "threshold": report.threshold,
                "confirmed_amounts": confirmed,
                "recomputed_amounts": recomputed,
                "confirmed_audit_list": list(report.audit_list),
                "recomputed_audit_list": recomputed_audit,
                "matches": matches,
            }


# ============================================================
# 序列化
# ============================================================


def batch_to_dict(batch: PartitionBatch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "period": batch.period,
        "parent_region": batch.parent_region,
        "version": batch.version,
        "status": batch.status,
        "supersedes": batch.supersedes,
        "created_seq": batch.created_seq,
        "total_amount": batch.total_amount,
        "partitions": [
            {
                "province": part.province,
                "amount": part.amount,
                "txn_count": len(part.txn_ids),
                "txn_ids": list(part.txn_ids),
            }
            for part in batch.partitions
        ],
    }


def report_to_dict(report: DiffReport) -> dict[str, Any]:
    return {
        "report_id": report.report_id,
        "batch_id": report.batch_id,
        "batch_version": report.batch_version,
        "period": report.period,
        "parent_region": report.parent_region,
        "threshold": report.threshold,
        "total_amount": report.total_amount,
        "audit_total": report.audit_total,
        "audit_list": list(report.audit_list),
        "province_metrics": [
            {
                "province": m.province,
                "amount": m.amount,
                "txn_count": len(m.txn_ids),
                "txn_ids": list(m.txn_ids),
                "reaches_threshold": m.reaches_threshold,
            }
            for m in report.province_metrics
        ],
        "validated": report.validated,
        "validation_errors": list(report.validation_errors),
        "status": report.status,
        "created_seq": report.created_seq,
        "confirmed_seq": report.confirmed_seq,
    }


# ============================================================
# 演示数据：华东/华南经营归属与成交行
# ============================================================


def build_demo_service() -> DrilldownAttributionService:
    """构造演示服务：广东 2026-10 起由华东划入华南，主场景落在 2026-10。"""
    assignments = [
        RegionAssignment("江苏", "华东", "2026-01"),
        RegionAssignment("上海", "华东", "2026-01"),
        RegionAssignment("广东", "华东", "2026-01", "2026-10"),  # 2026-10 起划出
        RegionAssignment("广东", "华南", "2026-10"),
        RegionAssignment("福建", "华南", "2026-01"),
    ]
    transactions: list[TransactionRow] = []
    # 2026-10：江苏 6×10=60，上海 4×10=40（华东合计 100）；广东 20×10=200 属华南
    for i in range(6):
        transactions.append(TransactionRow(f"T-10JS-{i + 1:02d}", "2026-10", "江苏", 10.0))
    for i in range(4):
        transactions.append(TransactionRow(f"T-10SH-{i + 1:02d}", "2026-10", "上海", 10.0))
    for i in range(20):
        transactions.append(TransactionRow(f"T-10GD-{i + 1:02d}", "2026-10", "广东", 10.0))
    # 2026-09：江苏 3×10=30，广东 5×10=50（当时广东仍属华东，合计 80）
    for i in range(3):
        transactions.append(TransactionRow(f"T-09JS-{i + 1:02d}", "2026-09", "江苏", 10.0))
    for i in range(5):
        transactions.append(TransactionRow(f"T-09GD-{i + 1:02d}", "2026-09", "广东", 10.0))
    return DrilldownAttributionService(assignments, transactions)
