"""维度下钻经营归属：分区明细批次 + 差异报告验收测试。

验收场景：同一时段华东 = 江苏 60 元 + 上海 40 元 = 100 元，广东 200 元属华南；
打开华东下钻明细后、报告确认前，迟到更正把江苏的 10 元移给上海，两个版本
总额相同，但实际超过 55 元核查门槛的省份不同（第一轮 [江苏]，第二轮 []）。
分区和金额必须来自同一版本，并发确认同一报告只能出数一次，历史报告可复算。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.drilldown import get_service
from app.services.drilldown_attribution import (
    AlreadyConfirmedError,
    CorrectionError,
    DrilldownAttributionService,
    PartitionNotFoundError,
    RegionAssignment,
    ReportStateError,
    StaleBatchError,
    TransactionRow,
    build_demo_service,
)

PERIOD = "2026-10"
PARENT = "华东"
THRESHOLD = 55.0


@pytest.fixture
def service() -> DrilldownAttributionService:
    """每个用例一份隔离的演示数据服务。"""
    return build_demo_service()


def _amounts(batch) -> dict[str, float]:
    return {part.province: part.amount for part in batch.partitions}


def _open_huadong(service: DrilldownAttributionService):
    return service.open_batch(PERIOD, PARENT)


def _correct_js_to_sh(service: DrilldownAttributionService, batch_id: str):
    """迟到更正：把江苏的一笔 10 元成交行移给上海。"""
    return service.apply_correction(
        batch_id, [{"txn_id": "T-10JS-01", "to_province": "上海"}])


# ============================================================
# 当时有效的经营区域
# ============================================================

class TestEffectiveRegions:
    def test_children_only_from_effective_regions(self, service):
        # 2026-10 广东已划入华南，华东只剩江苏、上海
        assert service.effective_children(PERIOD, PARENT) == ["上海", "江苏"]
        assert service.effective_children(PERIOD, "华南") == ["广东", "福建"]

    def test_children_follow_attribution_validity_window(self, service):
        # 2026-09 广东仍属华东：子类只用当时有效的经营区域
        assert service.effective_children("2026-09", PARENT) == ["上海", "广东", "江苏"]
        assert service.effective_children("2026-09", "华南") == ["福建"]

    def test_region_tree_by_period(self, service):
        tree = service.region_tree(PERIOD)
        assert tree["华东"] == ["上海", "江苏"]
        assert tree["华南"] == ["广东", "福建"]


# ============================================================
# 分区明细批次：按时段+父层查询经营归属
# ============================================================

class TestPartitionBatch:
    def test_huadong_total_is_exactly_100(self, service):
        batch = _open_huadong(service)
        assert _amounts(batch) == {"江苏": 60.0, "上海": 40.0}
        assert batch.total_amount == 100.0
        assert batch.version == 1
        assert batch.status == "有效"

    def test_guangdong_200_belongs_to_huanan_not_huadong(self, service):
        huadong = _open_huadong(service)
        assert "广东" not in _amounts(huadong)
        huanan = service.open_batch(PERIOD, "华南")
        assert _amounts(huanan) == {"广东": 200.0}
        assert huanan.total_amount == 200.0

    def test_open_is_idempotent_per_period_and_parent(self, service):
        first = _open_huadong(service)
        second = _open_huadong(service)
        assert first.batch_id == second.batch_id
        assert second.version == 1

    def test_historical_period_uses_then_effective_regions(self, service):
        # 2026-09 广东仍属华东：华东批次含广东 50 元
        batch = service.open_batch("2026-09", PARENT)
        assert _amounts(batch) == {"江苏": 30.0, "广东": 50.0}
        assert batch.total_amount == 80.0

    def test_partition_links_original_transaction_rows(self, service):
        batch = _open_huadong(service)
        js = next(p for p in batch.partitions if p.province == "江苏")
        assert js.txn_ids == tuple(f"T-10JS-{i:02d}" for i in range(1, 7))
        assert sum(10.0 for _ in js.txn_ids) == js.amount == 60.0


# ============================================================
# 省份页 + 完整父层：同一份数据
# ============================================================

class TestProvincePageAndParentLayer:
    def test_province_page_paginates_transaction_rows(self, service):
        batch = _open_huadong(service)
        page1 = service.province_transactions(batch.batch_id, "江苏", page=1, size=4)
        page2 = service.province_transactions(batch.batch_id, "江苏", page=2, size=4)
        assert page1["total_rows"] == 6
        assert page1["partition_amount"] == 60.0
        assert len(page1["items"]) == 4
        assert len(page2["items"]) == 2
        ids = [row["txn_id"] for row in page1["items"] + page2["items"]]
        assert ids == [f"T-10JS-{i:02d}" for i in range(1, 7)]

    def test_report_amounts_match_province_pages_and_parent_layer(self, service):
        """查询一个省份页和完整父层，再从同一份数据真正逐省计算报告金额。"""
        batch = _open_huadong(service)
        # 省份页逐行求和
        page_sums = {}
        for province in ("江苏", "上海"):
            rows = service.province_transactions(
                batch.batch_id, province, page=1, size=200)
            page_sums[province] = round(
                sum(row["amount"] for row in rows["items"]), 2)
        # 差异报告逐省计算
        report = service.compute_diff_report(batch.batch_id, THRESHOLD)
        report_amounts = {m.province: m.amount for m in report.province_metrics}
        assert page_sums == {"江苏": 60.0, "上海": 40.0}
        assert report_amounts == page_sums == _amounts(batch)
        assert report.total_amount == 100.0

    def test_province_not_in_batch_rejected(self, service):
        batch = _open_huadong(service)
        with pytest.raises(PartitionNotFoundError, match="广东"):
            service.province_transactions(batch.batch_id, "广东")


# ============================================================
# 迟到更正：新版本批次，总额不变
# ============================================================

class TestLateCorrection:
    def test_correction_creates_new_version_with_same_total(self, service):
        v1 = _open_huadong(service)
        v2 = _correct_js_to_sh(service, v1.batch_id)
        assert v2.version == 2
        assert v2.supersedes == v1.batch_id
        assert _amounts(v2) == {"江苏": 50.0, "上海": 50.0}
        # 两个版本的总额相同
        assert v2.total_amount == v1.total_amount == 100.0
        # 旧批次标记已更正，且内容保持原样（历史可查）
        assert v1.status == "已更正"
        assert _amounts(v1) == {"江苏": 60.0, "上海": 40.0}

    def test_correction_order_is_traceable(self, service):
        v1 = _open_huadong(service)
        v2 = _correct_js_to_sh(service, v1.batch_id)
        assert v1.created_seq < v2.created_seq
        log = service.correction_log()
        assert len(log) == 1
        assert log[0]["from_batch"] == v1.batch_id
        assert log[0]["to_batch"] == v2.batch_id
        assert log[0]["moves"] == [{
            "txn_id": "T-10JS-01",
            "from_province": "江苏",
            "to_province": "上海",
            "amount": 10.0,
        }]

    def test_correction_moves_specific_transaction_row(self, service):
        v1 = _open_huadong(service)
        v2 = _correct_js_to_sh(service, v1.batch_id)
        sh = next(p for p in v2.partitions if p.province == "上海")
        assert "T-10JS-01" in sh.txn_ids
        # 省份页能看到该行的原始归属（江苏）与本版归属（上海）
        page = service.province_transactions(v2.batch_id, "上海", page=1, size=200)
        moved = next(r for r in page["items"] if r["txn_id"] == "T-10JS-01")
        assert moved["province"] == "江苏"
        assert moved["assigned_province"] == "上海"

    def test_cannot_correct_a_superseded_batch(self, service):
        v1 = _open_huadong(service)
        _correct_js_to_sh(service, v1.batch_id)
        with pytest.raises(StaleBatchError):
            service.apply_correction(
                v1.batch_id, [{"txn_id": "T-10JS-02", "to_province": "上海"}])

    def test_correction_target_must_be_effective_child(self, service):
        v1 = _open_huadong(service)
        # 广东在 2026-10 不属于华东当时有效的经营区域
        with pytest.raises(CorrectionError, match="当时有效"):
            service.apply_correction(
                v1.batch_id, [{"txn_id": "T-10JS-01", "to_province": "广东"}])

    def test_correction_rejects_unknown_or_noop_moves(self, service):
        v1 = _open_huadong(service)
        with pytest.raises(CorrectionError):
            service.apply_correction(
                v1.batch_id, [{"txn_id": "T-404", "to_province": "上海"}])
        with pytest.raises(CorrectionError):
            service.apply_correction(
                v1.batch_id, [{"txn_id": "T-10JS-01", "to_province": "江苏"}])


# ============================================================
# 差异报告：逐省计算、核查名单、版本一致
# ============================================================

class TestDiffReport:
    def test_first_round_audit_list_before_correction(self, service):
        v1 = _open_huadong(service)
        report = service.compute_diff_report(v1.batch_id, THRESHOLD)
        assert report.batch_version == 1
        assert {m.province: m.amount for m in report.province_metrics} == {
            "江苏": 60.0, "上海": 40.0}
        assert report.total_amount == 100.0
        # 第一轮实际超过 55 元的省份：江苏
        assert list(report.audit_list) == ["江苏"]
        assert report.audit_total == 60.0
        assert report.validated and not report.validation_errors

    def test_two_rounds_of_audit_lists_differ(self, service):
        """核对更正先后及两轮实际需要核查的省份。"""
        v1 = _open_huadong(service)
        r1 = service.compute_diff_report(v1.batch_id, THRESHOLD)
        v2 = _correct_js_to_sh(service, v1.batch_id)
        r2 = service.compute_diff_report(v2.batch_id, THRESHOLD)
        # 更正先后：r1 基于 v1 计算，随后才产生 v2 与 r2
        assert r1.created_seq < v2.created_seq < r2.created_seq
        assert list(r1.audit_list) == ["江苏"]  # 60 >= 55
        assert list(r2.audit_list) == []  # 50、50 均未达 55
        assert r1.total_amount == r2.total_amount == 100.0

    def test_report_traces_transaction_rows_per_province(self, service):
        v1 = _open_huadong(service)
        report = service.compute_diff_report(v1.batch_id, THRESHOLD)
        js = next(m for m in report.province_metrics if m.province == "江苏")
        assert js.txn_ids == tuple(f"T-10JS-{i:02d}" for i in range(1, 7))
        assert js.reaches_threshold is True

    def test_no_cross_version_mixing(self, service):
        """分区和已计算的金额必须来自同一版本：旧江苏 60 不能加到新上海 50。"""
        v1 = _open_huadong(service)
        r1 = service.compute_diff_report(v1.batch_id, THRESHOLD)
        v2 = _correct_js_to_sh(service, v1.batch_id)
        r2 = service.compute_diff_report(v2.batch_id, THRESHOLD)
        for report in (r1, r2):
            amounts = {m.province: m.amount for m in report.province_metrics}
            assert round(sum(amounts.values()), 2) == report.total_amount == 100.0
        # 任何报告都不会出现 江苏60 + 上海50 = 110 的混版合计
        assert r1.total_amount != 110.0
        assert r2.total_amount != 110.0
        assert {m.province: m.amount for m in r1.province_metrics} == {
            "江苏": 60.0, "上海": 40.0}
        assert {m.province: m.amount for m in r2.province_metrics} == {
            "江苏": 50.0, "上海": 50.0}


# ============================================================
# 确认出数：幂等 + 版本校验；历史报告可复算
# ============================================================

class TestConfirmAndRecalculate:
    def test_confirm_after_correction_requires_current_version(self, service):
        """打开明细后、确认前发生迟到更正：不能混用更正前后的归属出数。"""
        v1 = _open_huadong(service)
        r1 = service.compute_diff_report(v1.batch_id, THRESHOLD)
        _correct_js_to_sh(service, v1.batch_id)
        with pytest.raises(StaleBatchError, match="重算"):
            service.confirm_report(r1.report_id)
        assert r1.status == "待确认"

    def test_confirm_once_and_only_once(self, service):
        v1 = _open_huadong(service)
        report = service.compute_diff_report(v1.batch_id, THRESHOLD)
        confirmed = service.confirm_report(report.report_id)
        assert confirmed.status == "已确认"
        assert confirmed.confirmed_seq is not None
        with pytest.raises(AlreadyConfirmedError):
            service.confirm_report(report.report_id)

    def test_concurrent_confirm_outputs_exactly_once(self, service):
        v1 = _open_huadong(service)
        report = service.compute_diff_report(v1.batch_id, THRESHOLD)

        def attempt():
            try:
                service.confirm_report(report.report_id)
                return "ok"
            except AlreadyConfirmedError:
                return "duplicate"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: attempt(), range(16)))
        assert results.count("ok") == 1
        assert results.count("duplicate") == 15
        assert report.status == "已确认"

    def test_historical_report_recalculates_from_pinned_version(self, service):
        v1 = _open_huadong(service)
        v2 = _correct_js_to_sh(service, v1.batch_id)
        r2 = service.compute_diff_report(v2.batch_id, THRESHOLD)
        service.confirm_report(r2.report_id)
        # 确认后又发生新的更正（v3：江苏再移 10 元给上海）
        service.apply_correction(
            v2.batch_id, [{"txn_id": "T-10JS-02", "to_province": "上海"}])
        # 历史报告仍按其锁定的 v2 复算，结果一致
        outcome = service.recalculate_report(r2.report_id)
        assert outcome["batch_version"] == 2
        assert outcome["batch_status"] == "已更正"
        assert outcome["recomputed_amounts"] == {"江苏": 50.0, "上海": 50.0}
        assert outcome["recomputed_audit_list"] == []
        assert outcome["matches"] is True

    def test_recalculate_requires_confirmed_report(self, service):
        v1 = _open_huadong(service)
        report = service.compute_diff_report(v1.batch_id, THRESHOLD)
        with pytest.raises(ReportStateError):
            service.recalculate_report(report.report_id)


# ============================================================
# HTTP 接口冒烟：完整业务流
# ============================================================

class TestHttpApi:
    @pytest.fixture
    def client(self):
        service = build_demo_service()
        app.dependency_overrides[get_service] = lambda: service
        try:
            yield TestClient(app), service
        finally:
            app.dependency_overrides.pop(get_service, None)

    def test_full_drilldown_flow(self, client):
        http, _ = client
        # 当时有效的经营区域
        resp = http.get("/api/drilldown/regions", params={"period": PERIOD})
        assert resp.status_code == 200
        tree = {p["parent_region"]: p["children"] for p in resp.json()["parents"]}
        assert tree["华东"] == ["上海", "江苏"]

        # 打开华东下钻明细：分区明细批次
        resp = http.get("/api/drilldown/partition-batches",
                        params={"period": PERIOD, "parent": PARENT})
        assert resp.status_code == 200
        batch = resp.json()
        assert batch["total_amount"] == 100.0
        assert {p["province"]: p["amount"] for p in batch["partitions"]} == {
            "江苏": 60.0, "上海": 40.0}

        # 省份页：江苏合计所用的成交行
        resp = http.get(
            f"/api/drilldown/partition-batches/{batch['batch_id']}"
            "/provinces/江苏/transactions",
            params={"page": 1, "size": 4})
        assert resp.status_code == 200
        page = resp.json()
        assert page["total_rows"] == 6
        assert page["partition_amount"] == 60.0

        # 计算差异报告（第一轮：核查名单 [江苏]）
        resp = http.post("/api/drilldown/diff-reports",
                         json={"batch_id": batch["batch_id"], "threshold": THRESHOLD})
        assert resp.status_code == 201
        r1 = resp.json()
        assert r1["audit_list"] == ["江苏"]
        assert r1["total_amount"] == 100.0

        # 迟到更正：江苏 10 元移给上海
        resp = http.post(
            f"/api/drilldown/partition-batches/{batch['batch_id']}/corrections",
            json={"moves": [{"txn_id": "T-10JS-01", "to_province": "上海"}]})
        assert resp.status_code == 201
        v2 = resp.json()
        assert v2["version"] == 2
        assert v2["total_amount"] == 100.0
        assert {p["province"]: p["amount"] for p in v2["partitions"]} == {
            "江苏": 50.0, "上海": 50.0}

        # 旧版本报告确认被拒：不能混用更正前后的归属
        resp = http.post(f"/api/drilldown/diff-reports/{r1['report_id']}/confirm")
        assert resp.status_code == 409

        # 基于新版本重算（第二轮：核查名单为空）并确认出数
        resp = http.post("/api/drilldown/diff-reports",
                         json={"batch_id": v2["batch_id"], "threshold": THRESHOLD})
        assert resp.status_code == 201
        r2 = resp.json()
        assert r2["audit_list"] == []
        resp = http.post(f"/api/drilldown/diff-reports/{r2['report_id']}/confirm")
        assert resp.status_code == 200
        assert resp.json()["status"] == "已确认"

        # 同一报告只能出数一次
        resp = http.post(f"/api/drilldown/diff-reports/{r2['report_id']}/confirm")
        assert resp.status_code == 409

        # 历史报告可复算
        resp = http.post(f"/api/drilldown/diff-reports/{r2['report_id']}/recalculate")
        assert resp.status_code == 200
        outcome = resp.json()
        assert outcome["matches"] is True
        assert outcome["recomputed_amounts"] == {"江苏": 50.0, "上海": 50.0}

    def test_unknown_batch_returns_404(self, client):
        http, _ = client
        resp = http.get("/api/drilldown/partition-batches/PB-9999")
        assert resp.status_code == 404
