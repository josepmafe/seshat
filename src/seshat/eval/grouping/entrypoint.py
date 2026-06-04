from __future__ import annotations

import asyncio

from seshat.agents.identification.grouping import GroupingAgent
from seshat.config.eval_settings import EvalConfig
from seshat.config.settings import SeshatConfig
from seshat.eval.grouping.runner import GroupingEvalRunner
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.pipeline.llm_factory import get_identification_llm
from seshat.utils.log import get_logger

logger = get_logger(__name__)


async def run(config: EvalConfig, seshat_config: SeshatConfig) -> None:
    setup_mlflow(config.observability, disable_autolog=True)

    llm = get_identification_llm(seshat_config)
    agent = GroupingAgent(llm=llm, config=seshat_config.extraction.identification)
    runner = GroupingEvalRunner(agent=agent, config=config)
    gate = await runner.run()
    logger.info("grouping eval: passed=%s", gate.passed)


if __name__ == "__main__":
    seshat_config = SeshatConfig()
    eval_config = EvalConfig()
    asyncio.run(run(eval_config, seshat_config))
