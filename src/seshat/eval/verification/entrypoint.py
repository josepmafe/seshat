from __future__ import annotations

import asyncio

from seshat.agents.verification import VerificationAgent
from seshat.config.eval_settings import EvalConfig
from seshat.config.settings import SeshatConfig
from seshat.eval.verification.runner import VerificationEvalRunner
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.pipeline.llm_factory import get_verification_llm
from seshat.utils.log import get_logger

logger = get_logger(__name__)


async def run(config: EvalConfig, seshat_config: SeshatConfig, model_id: str | None = None) -> None:
    setup_mlflow(config.observability, disable_autolog=True)

    llm = get_verification_llm(seshat_config)
    agent = VerificationAgent(llm=llm, config=seshat_config.extraction.verification)
    runner = VerificationEvalRunner(agent=agent, config=config, model_id=model_id)
    gate = await runner.run()
    logger.info("verification eval: passed=%s", gate.passed)


if __name__ == "__main__":
    seshat_config = SeshatConfig()
    eval_config = EvalConfig()
    asyncio.run(run(eval_config, seshat_config))
