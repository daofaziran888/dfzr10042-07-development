"""业务路由汇总：通用资源路由按模块分派，维度下钻经营归属走独立领域路由。"""
from __future__ import annotations

from app.routers import drilldown, resource

# drilldown 的显式路径先于 resource 的 /{module} 动态路径注册，避免歧义。
ROUTERS = [drilldown, resource]
