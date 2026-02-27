import os, json
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from .provider_base import AIProvider, SuggestRulesResponse, ExplainAnomalyResponse, DetectDriftResponse

class FoundryAgentProvider(AIProvider):
    def __init__(self, agent_id: str):
        self.client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential()
        )
        self.agent_id = agent_id

    def _ask_agent(self, prompt: str) -> str:
        # Pattern from Foundry Agent Service concepts: thread -> message -> run -> read messages :contentReference[oaicite:4]{index=4}
        thread = self.client.agents.create_thread()
        self.client.agents.create_message(thread_id=thread.id, role="user", content=prompt)
        run = self.client.agents.create_run(thread_id=thread.id, agent_id=self.agent_id)

        # poll until completed
        run = self.client.agents.get_run(thread_id=thread.id, run_id=run.id, poll=True)

        # fetch latest assistant message
        msgs = self.client.agents.list_messages(thread_id=thread.id)
        assistant = next(m for m in reversed(msgs.data) if m.role == "assistant")
        return assistant.content[0].text.value if assistant.content else ""

    def suggest_rules(self, dataset_id, profiling, existing_ruleset_yaml=None):
        prompt = build_prompt_suggest_rules(dataset_id, profiling, existing_ruleset_yaml)
        txt = self._ask_agent(prompt)
        # expectation: YAML + short rationale (or JSON)
        return SuggestRulesResponse(ruleset_yaml=extract_yaml(txt), rationale=extract_rationale(txt))

    def explain_anomaly(self, dataset_id, dq_failures, profiling):
        prompt = build_prompt_explain(dataset_id, dq_failures, profiling)
        txt = self._ask_agent(prompt)
        obj = json.loads(extract_json(txt))
        return ExplainAnomalyResponse(**obj)

    def detect_drift(self, dataset_id, profiling_now, profiling_baseline):
        prompt = build_prompt_drift(dataset_id, profiling_now, profiling_baseline)
        txt = self._ask_agent(prompt)
        obj = json.loads(extract_json(txt))
        return DetectDriftResponse(**obj)