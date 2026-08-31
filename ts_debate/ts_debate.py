import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_openai import ChatOpenAI
from .agent_prompts import  ModalityType
from .agent_specs import JudgeAgent, KnowledgeElicitor, NumericalAgent, TextAgent, VisualAgent
from utils.llm_providers import CostMonitor, create_chat_model
from utils.numerical_lookup import NumericalLookupFunction
from utils.task_config import fill_real_context


# EVIDENCE STATE
@dataclass
class DebateMessage:
    """A message in the evidence presentation."""

    agent_id: str
    modality: ModalityType
    content: str
    round_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateState:
    """State of the evidence presentation."""

    task_description: str
    context: Dict[str, Any]
    messages: List[DebateMessage] = field(default_factory=list)
    current_round: int = 1
    max_rounds: int = 2  # Evidence + Refinement rounds
    final_decision: Optional[str] = None

    def add_message(self, message: DebateMessage):
        self.messages.append(message)

    def get_full_history(self) -> str:
        """Get all evidence presentations."""
        if not self.messages:
            return "No evidence presentations."
        return "\n\n".join([
            f"[{m.agent_id}]: {m.content}"
            for m in self.messages
        ])

    def get_round_history(self, round_num: int) -> str:
        """Get history up to (but not including) a specific round."""
        prev_messages = [
            m for m in self.messages if m.round_number < round_num]
        if not prev_messages:
            return "No previous statements."
        return "\n\n".join([
            f"[{m.agent_id}]: {m.content}"
            for m in prev_messages
        ])

    def get_final_round(self) -> str:
        """Get ONLY the final round's messages (for judges)."""
        if not self.messages:
            return "No evidence presentations."
        max_round = max(m.round_number for m in self.messages)
        final_messages = [
            m for m in self.messages if m.round_number == max_round]
        return "\n\n".join([
            f"[{m.agent_id}]: {m.content}"
            for m in final_messages
        ])

    def get_previous_round(self, round_num: int) -> str:
        """Get ONLY the most recent round's messages (for refinement)."""
        if round_num <= 1:
            return ""
        prev_round = round_num - 1
        prev_messages = [
            m for m in self.messages if m.round_number == prev_round]
        if not prev_messages:
            return ""
        return "\n\n".join([
            f"[{m.agent_id}]: {m.content}"
            for m in prev_messages
        ])


# MULTI-MODAL EVIDENCE SYNTHESIS ORCHESTRATOR
class CrossModalDebateOrchestrator:
    """
    Orchestrates evidence presentation from modality-specialized agents.

    Design:
    - Round 1: Each modality presents their evidence independently
    - Round 2+: Each modality refines based on cross-modal insights
    - Judges score evidence quality and synthesize weighted answers
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        monitor: Optional[CostMonitor] = None,
        num_judges: int = 3,
        max_rounds: int = 2,  # Evidence + Refinement rounds
        max_judge_rounds: int = 1,
        verbose: bool = True,
        # Component ablation flags
        use_lookup: bool = True,
        judge_use_code_executor: bool = True,
        judge_use_lookup: bool = True,
        judge_only: bool = False,
        # Modality ablation flags
        use_text_agents: bool = True,
        use_visual_agents: bool = True,
        use_numerical_agents: bool = True,
        # Frequency features flag
        use_frequency_features: bool = True,
        # Judge chart restriction flags
        judge_use_time_chart: bool = True,
        judge_use_freq_chart: bool = True,
        # Knowledge elicitation flag
        use_knowledge_elicitation: bool = True,
    ):
        """
        Initialize the evidence synthesis orchestrator.

        Args:
            provider: LLM provider ("openrouter")
            model: Model identifier
            api_key: API key for provider
            monitor: CostMonitor for tracking costs
            num_judges: Number of judge agents
            max_rounds: Number of evidence rounds (default 2: evidence + refinement)
            max_judge_rounds: Maximum rounds for judge deliberation
            verbose: Print progress
            use_lookup: Enable Lookup Function for Numerical agents
            judge_use_code_executor: Enable Code Executor for Judge
            judge_use_lookup: Enable Lookup Function for Judge
            use_knowledge_elicitation: Enable Stage 0 knowledge elicitation
        """
        self.provider = provider
        self.model_name = model
        self.api_key = api_key
        self.monitor = monitor
        self.num_judges = num_judges
        self.max_rounds = max_rounds
        self.max_judge_rounds = max_judge_rounds
        self.verbose = verbose
        # Component ablation flags
        self.use_lookup = use_lookup
        self.judge_use_code_executor = judge_use_code_executor
        self.judge_use_lookup = judge_use_lookup
        self.judge_only = judge_only
        # Modality ablation flags
        self.use_text_agents = use_text_agents
        self.use_visual_agents = use_visual_agents
        self.use_numerical_agents = use_numerical_agents
        # Frequency features flag
        self.use_frequency_features = use_frequency_features
        # Judge chart restriction flags
        self.judge_use_time_chart = judge_use_time_chart
        self.judge_use_freq_chart = judge_use_freq_chart
        # Knowledge elicitation flag
        self.use_knowledge_elicitation = use_knowledge_elicitation

    def _create_model(self) -> ChatOpenAI:
        """Create a new model instance."""
        return create_chat_model(
            provider=self.provider,
            model=self.model_name,
            api_key=self.api_key,
            monitor=self.monitor,
            framework="TS-Debate",
            temperature=0.0,
        )

    def run_debate(self, task_description: str, context: Dict[str, Any]) -> tuple[DebateState, str]:
        """
        Run the evidence presentation and synthesis.

        Args:
            task_description: The task prompt from task_config.py
            context: Context dict with:
                - text: Textual description
                - values: Numerical time series data
                - timestamps: Time points
                - time_series_chart: Base64 chart image
                - frequency_chart: Base64 frequency chart

        Returns:
            Tuple of (DebateState with final decision, filled_task_instruction)
        """
        state = DebateState(task_description=task_description,
                            context=context, max_rounds=self.max_rounds)

        # Store judge/synthesizer logs
        self.judge_traces: List[str] = []

        # Prepare lookup function
        values = context.get("values", [])
        if values and isinstance(values[0], (list, np.ndarray)):
            n_timesteps = len(values[0])
        else:
            n_timesteps = len(values) if values else 0
        timestamps = context.get("timestamps", list(range(n_timesteps)))
        indicator_values = context.get("input_indicator")
        indicator_label = context.get("indicator_label", "Indicator")
        lookup_fn = (
            NumericalLookupFunction(
                values=values,
                timestamps=timestamps,
                indicator_values=indicator_values,
                indicator_label=indicator_label,
                sample_metadata=context,
            )
            if values
            else None
        )

        # KE sees text context (to understand task) but not numbers (to prevent premature analysis)
        ke_task_description = fill_real_context(task_description, context, allow_number=False, allow_text=True)
        filled_instruction = fill_real_context(task_description, context)

        # Build list of available modalities
        available_modalities = []
        if self.use_text_agents:
            available_modalities.append("TEXT")
        if self.use_visual_agents:
            available_modalities.append("VISUAL")
        if self.use_numerical_agents:
            available_modalities.append("NUMERICAL")
        if not available_modalities:
            available_modalities = ["TEXT", "VISUAL", "NUMERICAL"]

        # KNOWLEDGE ELICITATION (Pre-Analysis)
        domain_knowledge = ""
        if self.use_knowledge_elicitation:
            if self.monitor:
                self.monitor.set_stage("knowledge_elicitation")

            if self.verbose:
                print(f"\n{'=' * 80}")
                print("KNOWLEDGE ELICITATION")
                print(f"{'=' * 80}")

            # Run knowledge elicitation (task-agnostic - works for all task types)
            elicitor = KnowledgeElicitor(
                llm=self._create_model(),
                task_description=ke_task_description,
            )
            _start = time.time()
            domain_knowledge = elicitor.elicit()
            if self.monitor:
                self.monitor.record_stage_time("knowledge_elicitation", time.time() - _start)

            # Store in context for reference
            context["domain_knowledge"] = domain_knowledge

            if self.verbose:
                print(f"\n{domain_knowledge}")

        # Judge-only mode
        if self.judge_only:
            if self.verbose:
                print(f"\n{'=' * 80}")
                print("TS-DEBATE: Reviewer-Only Mode (No Evidence Presentation)")
                print(f"{'=' * 80}")
            state.current_round = 0
            # Prepare task descriptions
            task_description_judge = fill_real_context(task_description, context, allow_number=not self.judge_use_lookup, allow_text=True)
            full_history = "No evidence presentations. Analyze the task directly using provided data."
        else:
            # Evidence Presentation Mode
            task_description_text = fill_real_context(
                task_description, context, allow_number=False, allow_text=True)
            task_description_visual = fill_real_context(
                task_description, context, allow_number=False, allow_text=False)
            task_description_numerical = fill_real_context(
                task_description, context, allow_number=not self.use_lookup, allow_text=False)
            # Prepare task descriptions
            task_description_judge = fill_real_context(task_description, context, allow_number=not self.judge_use_lookup, allow_text=self.use_text_agents)

            if self.verbose:
                print(f"\n{'=' * 80}")
                print("TS-DEBATE: Multi-Modal Collaborative Debate")
                print(f"{'=' * 80}")

            # Create ONE agent per modality
            text_agent = None
            visual_agent = None
            numerical_agent = None

            if self.use_text_agents:
                text_agent = TextAgent("TEXT", self._create_model())

            if self.use_visual_agents:
                visual_agent = VisualAgent(
                    "VISUAL", self._create_model())

            if self.use_numerical_agents and values:
                agent_lookup = lookup_fn if self.use_lookup else None
                numerical_agent = NumericalAgent(
                    "NUMERICAL",
                    self._create_model(),
                    agent_lookup,
                    use_frequency_features=self.use_frequency_features,
                )

            modality_agents = [a for a in [text_agent,
                                           visual_agent, numerical_agent] if a is not None]

            if not modality_agents:
                raise ValueError(
                    "At least one modality must be enabled. Got: "
                    f"use_text_agents={self.use_text_agents}, "
                    f"use_visual_agents={self.use_visual_agents}, "
                    f"use_numerical_agents={self.use_numerical_agents}"
                )

            # Get context for each modality
            time_chart = context.get("time_series_chart")
            freq_chart = context.get("frequency_chart")

            # Run evidence presentation round(s)
            for round_num in range(1, self.max_rounds + 1):
                state.current_round = round_num

                round_type = "EVIDENCE PRESENTATION" if round_num == 1 else f"REFINEMENT ROUND {round_num}"

                # Get ONLY the previous round's evidence (not full history)
                debate_history = state.get_previous_round(round_num)

                if self.verbose:
                    print(f"\n{'─' * 80}")
                    print(f"Round {round_num}/{self.max_rounds}: {round_type}")
                    print(f"{'─' * 80}")

                # Define response functions with debate history and domain knowledge
                def run_text_agent(agent, history=debate_history, rnd=round_num, max_rnd=self.max_rounds, knowledge=domain_knowledge):
                    stage = f"debate_r{rnd}_TEXT"
                    if self.monitor:
                        self.monitor.set_stage(stage, agent.agent_id)
                    _start = time.time()
                    response = agent.respond(task_description_text, debate_history=history,
                                             round_num=rnd, max_rounds=max_rnd, domain_knowledge=knowledge)
                    if self.monitor:
                        self.monitor.record_stage_time(stage, time.time() - _start)
                    return (agent.agent_id, agent.modality, response)

                def run_visual_agent(agent, history=debate_history, rnd=round_num, max_rnd=self.max_rounds, knowledge=domain_knowledge):
                    stage = f"debate_r{rnd}_VISUAL"
                    if self.monitor:
                        self.monitor.set_stage(stage, agent.agent_id)
                    _start = time.time()
                    response = agent.respond(
                        task_description_visual,
                        time_chart,
                        freq_chart,
                        debate_history=history,
                        round_num=rnd,
                        max_rounds=max_rnd,
                        domain_knowledge=knowledge,
                    )
                    if self.monitor:
                        self.monitor.record_stage_time(stage, time.time() - _start)
                    return (agent.agent_id, agent.modality, response)

                def run_numerical_agent(agent, history=debate_history, rnd=round_num, max_rnd=self.max_rounds, knowledge=domain_knowledge):
                    stage = f"debate_r{rnd}_NUMERICAL"
                    if self.monitor:
                        self.monitor.set_stage(stage, agent.agent_id)
                    _start = time.time()
                    response = agent.respond(task_description_numerical, debate_history=history,
                                             round_num=rnd, max_rounds=max_rnd, domain_knowledge=knowledge)
                    if self.monitor:
                        self.monitor.record_stage_time(stage, time.time() - _start)
                    return (agent.agent_id, agent.modality, response)

                # Run ALL modality agents in PARALLEL
                round_responses = []
                with ThreadPoolExecutor(max_workers=len(modality_agents)) as executor:
                    futures = []

                    for agent in modality_agents:
                        if agent.modality == ModalityType.TEXT:
                            futures.append(executor.submit(
                                run_text_agent, agent))
                        elif agent.modality == ModalityType.VISUAL:
                            futures.append(executor.submit(
                                run_visual_agent, agent))
                        elif agent.modality == ModalityType.NUMERICAL:
                            futures.append(executor.submit(
                                run_numerical_agent, agent))

                    for future in as_completed(futures):
                        agent_id, modality, response = future.result()
                        round_responses.append((agent_id, modality, response))

                # Add responses to state
                for agent_id, modality, response in round_responses:
                    state.add_message(DebateMessage(
                        agent_id, modality, response, round_num))
                    if self.verbose:
                        print(f"\n[{agent_id}]:\n{response}")

            # Only final round evidence for judges (refined/complete)
            full_history = state.get_final_round()

        # Judge evaluation
        if self.verbose:
            print(f"\n{'─' * 80}")
            print("REVIEWER EVALUATION & SYNTHESIS")
            print(f"{'─' * 80}")

        judge_lookup = lookup_fn if self.judge_use_lookup else None
        judges = [
            JudgeAgent(
                f"reviewer_{i}",
                self._create_model(),
                context,
                judge_lookup,
                use_code_executor=self.judge_use_code_executor,
                use_time_chart=self.judge_use_time_chart,
                use_freq_chart=self.judge_use_freq_chart,
                use_frequency_features=self.use_frequency_features,
                domain_knowledge=domain_knowledge,
                text_in_task=self.use_text_agents,  # Text not embedded in judge task description
            )
            for i in range(self.num_judges)
        ]

        # Track each judge's previous response for independent self-refinement
        judge_previous_responses: Dict[int, str] = {}  # judge_idx -> their previous response

        judge_responses = []
        for judge_round in range(1, self.max_judge_rounds + 1):
            if self.verbose:
                print(f"\n  Reviewer Round {judge_round}/{self.max_judge_rounds}")

            def run_judge(judge, judge_idx, prev_response="", jrnd=judge_round):
                stage = f"reviewer_r{jrnd}_{judge.agent_id}"
                if self.monitor:
                    self.monitor.set_stage(stage, judge.agent_id)
                # Build history: agent debate + judge's OWN previous response only
                if prev_response:
                    history_with_self = f"{full_history}\n\n---\nYour previous analysis (refine if needed):\n{prev_response}"
                else:
                    history_with_self = full_history

                _start = time.time()
                response = judge.respond(
                    task_description_judge,
                    history_with_self,
                    available_modalities=available_modalities,
                )
                if self.monitor:
                    self.monitor.record_stage_time(stage, time.time() - _start)
                return (judge_idx, judge.agent_id, response)

            round_responses = []
            with ThreadPoolExecutor(max_workers=self.num_judges) as executor:
                futures = []
                for idx, judge in enumerate(judges):
                    # Get this judge's OWN previous response (empty for round 1)
                    prev = judge_previous_responses.get(idx, "")
                    futures.append(executor.submit(run_judge, judge, idx, prev))

                for future in as_completed(futures):
                    judge_idx, judge_id, response = future.result()
                    round_responses.append((judge_idx, judge_id, response))
                    if self.verbose:
                        print(f"\n    [{judge_id}]:\n{response}")

            # Update previous responses for next round (sliding window - replace, not append)
            round_responses.sort(key=lambda x: x[0])
            for judge_idx, judge_id, response in round_responses:
                judge_previous_responses[judge_idx] = response  # Replace with latest
                judge_responses.append(f"[{judge_id}]: {response}")

        # Final Synthesis
        if self.monitor:
            self.monitor.set_stage("synthesizer")

        if self.verbose:
            print(f"\n{'─' * 80}")
            print("FINAL SYNTHESIS")
            print(f"{'─' * 80}")

        synthesizer_lookup = lookup_fn if self.judge_use_lookup else None
        synthesizer = JudgeAgent(
            "decision_maker",
            self._create_model(),
            context,
            synthesizer_lookup,
            use_code_executor=self.judge_use_code_executor,
            use_time_chart=self.judge_use_time_chart,
            use_freq_chart=self.judge_use_freq_chart,
            use_frequency_features=self.use_frequency_features,
            domain_knowledge=domain_knowledge,
            is_synthesizer=True,
            text_in_task=self.use_text_agents,  # Text not embedded in synthesizer task description
        )

        final_round_responses = judge_responses[-self.num_judges:]
        all_judge_responses = "\n\n".join(final_round_responses)

        # Build verification sources for synthesizer
        verification_sources = []
        if self.judge_use_lookup and lookup_fn:
            lookup_tools = ["get_info", "get_values",
                            "get_around", "get_features"]
            if self.use_frequency_features:
                lookup_tools.append("get_frequency_features")
            if lookup_fn.is_multivariate:
                lookup_tools.extend(["get_channel_values", "get_all_channels"])
            if lookup_fn.has_indicator:
                lookup_tools.append("get_indicator")
            verification_sources.append(
                f"NUMERICAL TOOLS: {', '.join(lookup_tools)}")
        if self.judge_use_code_executor:
            verification_sources.append("CODE: execute_code(code)")
        has_charts = self.judge_use_time_chart or self.judge_use_freq_chart
        if has_charts:
            chart_types = []
            if self.judge_use_time_chart:
                chart_types.append("time series")
            if self.judge_use_freq_chart:
                chart_types.append("frequency")
            verification_sources.append(
                f"CHARTS: {', '.join(chart_types)} (attached)")
        # Check if text was in context (meaning it's embedded in task description)
        if self.use_text_agents:
            verification_sources.append("TEXT: embedded in task description")

        # Add domain knowledge as a verification source if available
        if domain_knowledge:
            verification_sources.append("DOMAIN: knowledge in system prompt")

        # Build verification section focused on ANSWERS (not claims)
        if verification_sources:
            verification_note = f"""VERIFICATION SOURCES (max 3 tool calls when available):
{chr(10).join('- ' + s for s in verification_sources)}"""
            verification_output = """ANSWER VERIFICATION (use TASK TYPE from above, summarizing only outstanding concerns):
- If PAST-PRESENT task: Verify answers against data
- If FUTURE task: Past data describes history but CANNOT verify predictions - evaluate domain reasoning"""
        else:
            # No verification sources at all
            verification_note = """NOTE: No verification sources available. Evaluate reasoning quality and domain consistency."""
            verification_output = """ANSWER EVALUATION:
- Compare reasoning quality and domain consistency
- If FUTURE task: domain reasoning quality decides"""

        synthesis_prompt = f"""Task: {task_description_judge}

ANALYSTS: {', '.join(available_modalities)}

Reviewer Evaluations:
{all_judge_responses}

---

{verification_note}

OUTPUT FORMAT (keep everything concise):
TASK: <Restate what the task is asking in one sentence>
TASK TYPE: [FUTURE / PAST-PRESENT]

APPROACH CHECK:
- SUGGESTED: <from domain knowledge>
- USED: <by reviewers>
- Status: <always re-check even with UNANIMOUS answers> [CORRECT / MISMATCH]

REVIEWER SCORES:
- Reviewer 0: (Task: _/20, Evidence: _/20, Verification: _/20, Conflicts: _/20, Calibration: _/20) = [0-100]
- Reviewer 1: (Task: _/20, Evidence: _/20, Verification: _/20, Conflicts: _/20, Calibration: _/20) = [0-100]
Note that perfect score is IMPOSSIBLE. Never give 100 score to any reviewer. There are always flaws in the reviewers' reasoning.

{verification_output}
**IMPORTANT**: Never verify claims against data for FUTURE tasks. Past time series CANNOT verify future predictions. Always check the requested date against the date at which the data actually ends. 
Do NOT refer to past data calculation or verification to support the answer. Only past data vs. future data (expected values by YOUR prediction) can be used to support the answer.

CONFLICT STATUS:
- Reviewer Agreement: [UNANIMOUS / SPLIT / ALL_DIFFERENT]
- Approach Status: [ALL_CORRECT / ALL_WRONG / MIXED]
- Analyst Agreement: [From reviewer reports - did analysts conflict?]
- Resolution: [VERIFIED_RESOLUTION / UNRESOLVED / NO_CONFLICT / APPROACH_ERROR]

CALIBRATED REASONING (apply based on TASK TYPE, summarizing in 1-2 short sentences):
- FUTURE task: <evaluate domain reasoning quality - cannot verify with past data; if reviewers said verified for FUTURE task, they are wrong!>
- PAST-PRESENT task: <data verification result>
- If MISMATCH: <derive own answer using correct approach>
- If SPLIT + cannot resolve: <Choose CONSERVATIVE answer>

FINAL ANSWER: <your calibrated answer in exact, user-defined task format>"""

        _start = time.time()
        final_decision = synthesizer.respond(
            synthesis_prompt, "", as_final=True)
        if self.monitor:
            self.monitor.record_stage_time("synthesizer", time.time() - _start)

        self.judge_traces.append(
            f"[Decision Maker]\n\n{synthesis_prompt}\n\n[Response]\n{final_decision}")

        state.final_decision = final_decision

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"{state.final_decision}")
            print(f"{'=' * 80}\n")

        return (state, filled_instruction)


class TSDebate:
    """
    TS-Debate: Multi-Modal Evidence Synthesis for Time Series.

    Stage 0: Knowledge elicitation (activates domain knowledge before analysis).
    Stage 1: Each modality presents evidence independently (Round 1).
    Stage 1+: Modalities refine based on cross-modal insights (Round 2+).
    Stage 2: Judges score evidence quality and synthesize weighted answers.
    Stage 3: Meta-reasoner synthesizer derives final calibrated answer.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        monitor: Optional[CostMonitor] = None,
        num_judges: int = 3,
        max_rounds: int = 2,  # Evidence + Refinement rounds
        max_judge_rounds: int = 1,
        verbose: bool = True,
        # Ablation flags
        use_lookup: bool = True,
        judge_use_code_executor: bool = True,
        judge_use_lookup: bool = True,
        judge_only: bool = False,
        # Modality ablation flags
        use_text_agents: bool = True,
        use_visual_agents: bool = True,
        use_numerical_agents: bool = True,
        # Frequency features flag
        use_frequency_features: bool = True,
        # Judge chart restriction flags
        judge_use_time_chart: bool = True,
        judge_use_freq_chart: bool = True,
        # Knowledge elicitation flag
        use_knowledge_elicitation: bool = True,
    ):
        """Initialize TS-Debate with evidence synthesis orchestrator."""
        self.orchestrator = CrossModalDebateOrchestrator(
            provider=provider,
            model=model,
            api_key=api_key,
            monitor=monitor,
            num_judges=num_judges,
            max_rounds=max_rounds,
            max_judge_rounds=max_judge_rounds,
            verbose=verbose,
            use_lookup=use_lookup,
            judge_use_code_executor=judge_use_code_executor,
            judge_use_lookup=judge_use_lookup,
            judge_only=judge_only,
            use_text_agents=use_text_agents,
            use_visual_agents=use_visual_agents,
            use_numerical_agents=use_numerical_agents,
            use_frequency_features=use_frequency_features,
            judge_use_time_chart=judge_use_time_chart,
            judge_use_freq_chart=judge_use_freq_chart,
            use_knowledge_elicitation=use_knowledge_elicitation,
        )
        if verbose:
            modalities = []
            if use_text_agents:
                modalities.append("TEXT")
            if use_visual_agents:
                modalities.append("VISUAL")
            if use_numerical_agents:
                modalities.append("NUMERICAL")
            ke_status = "✓" if use_knowledge_elicitation else "✗"
            print(
                f"🔬 TS-Debate (Evidence Synthesis) initialized: {len(modalities)} modalities, "
                f"{max_rounds} round(s), {num_judges} judges, KE: {ke_status}"
            )

    def run_debate(self, task_description: str, context: Dict[str, Any]) -> tuple[DebateState, str]:
        """
        Run TS-Debate evidence synthesis.

        Args:
            task_description: Task prompt from task_config.py
            context: Dict with text, values, timestamps, charts

        Returns:
            Tuple of (DebateState, filled_task_instruction)
        """
        return self.orchestrator.run_debate(task_description, context)



