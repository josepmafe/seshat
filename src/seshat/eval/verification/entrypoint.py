from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.agents.verification import VerificationAgent
from seshat.eval.mlflow_logging import log_eval_model
from seshat.eval.verification.runner import VerificationEvalRunner
from seshat.pipeline.llm_factory import get_verification_llm
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig
    from seshat.eval.corpus_tags import CorpusTagFilter


logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None) -> None:
    assert seshat_config.extraction.verification is not None
    llm = get_verification_llm(seshat_config)
    llm_cfg = seshat_config.extraction.verification
    agent = VerificationAgent(llm=llm, config=llm_cfg)

    logger.info("LLM provider=%r model=%r temperature=%s", llm_cfg.provider.value, llm_cfg.model, llm_cfg.temperature)
    model_id = log_eval_model("seshat-verification-agent", inference_component=agent, llm_config=llm_cfg)

    runner = VerificationEvalRunner(agent=agent, config=eval_config)
    gate = await runner.run(tag_filter=tag_filter, model_id=model_id)

    logger.info("verification eval: passed=%s", gate.passed)
