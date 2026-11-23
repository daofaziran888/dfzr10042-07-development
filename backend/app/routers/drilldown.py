"""维度下钻经营归属接口：分区明细批次查询、迟到更正、差异报告计算与确认。

路由层只做参数透传和领域错误到状态码的映射，业务判断全部在服务层。
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.drilldown_attribution import (
    AlreadyConfirmedError,
    BatchNotFoundError,
    CorrectionError,
    DrilldownAttributionService,
    DrilldownError,
    PartitionNotFoundError,
    ReportNotFoundError,
    ReportStateError,
    StaleBatchError,
    batch_to_dict,
    build_demo_service,
    report_to_dict,
)

router = APIRouter(prefix="/api/drilldown", tags=["维度下钻经营归属"])

_service = build_demo_service()


def get_service() -> DrilldownAttributionService:
    """服务单例，测试可经 dependency_overrides 替换为隔离实例。"""
    return _service


class CorrectionMove(BaseModel):
    txn_id: str
    to_province: str


class CorrectionPayload(BaseModel):
    moves: list[CorrectionMove] = Field(min_length=1)
    remark: str | None = None


class DiffReportPayload(BaseModel):
    batch_id: str
    threshold: float = Field(gt=0)


_ERROR_STATUS: list[tuple[type[DrilldownError], int]] = [
    (BatchNotFoundError, 404),
    (PartitionNotFoundError, 404),
    (ReportNotFoundError, 404),
    (StaleBatchError, 409),
    (AlreadyConfirmedError, 409),
    (CorrectionError, 400),
    (ReportStateError, 400),
]


def _call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return func(*args, **kwargs)
    except DrilldownError as exc:
        status = next(
            (code for err_type, code in _ERROR_STATUS if isinstance(exc, err_type)),
            400,
        )
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get("/regions")
def region_tree(
    period: str = Query(..., description="时段，如 2026-10"),
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """某时段各父层当时有效的经营区域（子地区）。"""
    tree = service.region_tree(period)
    return {
        "period": period,
        "parents": [
            {"parent_region": parent, "children": children}
            for parent, children in tree.items()
        ],
    }


@router.get("/partition-batches")
def open_batch(
    period: str = Query(...),
    parent: str = Query(..., description="父层经营区域，如 华东"),
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """按时段+父层打开下钻明细：返回当前有效的分区明细批次。"""
    return batch_to_dict(_call(service.open_batch, period, parent))


@router.get("/partition-batches/{batch_id}")
def get_batch(
    batch_id: str,
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """批次详情：分区、合计、版本与状态。"""
    return batch_to_dict(_call(service.get_batch, batch_id))


@router.get("/partition-batches/{batch_id}/provinces/{province}/transactions")
def province_transactions(
    batch_id: str,
    province: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=200),
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """省份页：该省分区合计所用的原始成交行（分页）。"""
    return _call(
        service.province_transactions, batch_id, province, page=page, size=size)


@router.post("/partition-batches/{batch_id}/corrections", status_code=201)
def apply_correction(
    batch_id: str,
    payload: CorrectionPayload,
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """迟到更正：在当前有效批次上改派成交行归属，生成新版本批次。"""
    moves = [move.model_dump() for move in payload.moves]
    return batch_to_dict(_call(service.apply_correction, batch_id, moves))


@router.get("/corrections")
def correction_log(
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """更正流水：按发生顺序返回，供核对更正先后。"""
    return {"items": service.correction_log()}


@router.post("/diff-reports", status_code=201)
def compute_diff_report(
    payload: DiffReportPayload,
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """用分区明细批次逐省计算差异报告（实际合计 + 核查名单）。"""
    report = _call(service.compute_diff_report, payload.batch_id, payload.threshold)
    return report_to_dict(report)


@router.get("/diff-reports/{report_id}")
def get_diff_report(
    report_id: str,
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """报告详情：含各省合计所用的成交行关联。"""
    return report_to_dict(_call(service.get_report, report_id))


@router.post("/diff-reports/{report_id}/confirm")
def confirm_diff_report(
    report_id: str,
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """确认出数：幂等，同一报告只能出数一次；版本被更正则拒绝。"""
    return report_to_dict(_call(service.confirm_report, report_id))


@router.post("/diff-reports/{report_id}/recalculate")
def recalculate_diff_report(
    report_id: str,
    service: DrilldownAttributionService = Depends(get_service),
) -> dict[str, Any]:
    """复算历史报告：按报告锁定的批次版本重算并核对。"""
    return _call(service.recalculate_report, report_id)
