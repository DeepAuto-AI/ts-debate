from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .agent_prompts import (
    EVIDENCE_RULES,
    JUDGE_EVALUATION_CRITERIA,
    JUDGE_PROTOCOL,
    KNOWLEDGE_CONTEXT_TEMPLATE,
    KNOWLEDGE_ELICITATION_PROMPT,
    KNOWLEDGE_ELICITOR_SYSTEM,
    MODALITY_PROFILES,
    NUMERICAL_AGENT_SYSTEM,
    SYNTHESIZER_PROTOCOL,
    TEMPORAL_AWARENESS,
    TEMPORAL_BASICS,
    TEXT_AGENT_SYSTEM,
    VISUAL_AGENT_SYSTEM,
    ModalityType,
    get_evidence_format,
    get_refinement_format,
)
from .agent_tools import create_code_executor_tool, create_lookup_tools
from utils.numerical_lookup import NumericalLookupFunction



class KnowledgeElicitor:
    """
    Elicits domain knowledge BEFORE any data analysis.
    Single call at Stage 0, shared across all agents/judges/synthesizer.

    This activates the LLM's pretrained domain knowledge to guide analysis
    and prevent blind inferences that contradict domain constraints.
    """

    def __init__(self, llm: ChatOpenAI, task_description: str):
        """
        Initialize knowledge elicitor.

        Args:
            llm: Language model to use for elicitation
            task_description: The task prompt (includes task type info)
        """
        self.llm = llm
        self.task_description = task_description

    def elicit(self) -> str:
        """
        Elicit domain knowledge for this task.

        Returns:
            String containing elicited domain knowledge in structured format.
        """
        # Build user prompt - task-agnostic, works for classification/regression/QA
        user_prompt = KNOWLEDGE_ELICITATION_PROMPT.format(
            task_description=self.task_description)

        # Call LLM with system + user messages
        messages = [
            SystemMessage(content=KNOWLEDGE_ELICITOR_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        content = response.content

        return content if isinstance(content, str) else str(content)

    @staticmethod
    def format_knowledge_context(domain_knowledge: str) -> str:
        """
        Format domain knowledge for injection into agent/judge/synthesizer prompts.

        Args:
            domain_knowledge: Raw elicited knowledge string

        Returns:
            Formatted knowledge context string
        """
        return KNOWLEDGE_CONTEXT_TEMPLATE.format(domain_knowledge=domain_knowledge)


class TextAgent:
    """TEXT modality agent for evidence presentation."""

    def __init__(self, agent_id: str, model: ChatOpenAI):
        self.agent_id = agent_id
        self.model = model
        self.modality = ModalityType.TEXT

        # Build system prompt (static role/behavioral guidance)
        self.system_prompt = TEXT_AGENT_SYSTEM.format(
            modality_profile=MODALITY_PROFILES[ModalityType.TEXT],
            temporal_basics=TEMPORAL_BASICS,
            evidence_rules=EVIDENCE_RULES,
        )

    def respond(
        self,
        task: str,
        debate_history: str = "",
        round_num: int = 1,
        max_rounds: int = 1,
        other_modalities: Optional[List[str]] = None,
        domain_knowledge: str = "",
    ) -> str:
        """Generate evidence presentation."""
        # Use refinement format for round 2+
        if round_num > 1 and debate_history:
            output_format = get_refinement_format(self.modality.name)
            history_section = f"""
=== PREVIOUS ROUND EVIDENCE ===
{debate_history}
=== END PREVIOUS EVIDENCE ===
"""
        else:
            output_format = get_evidence_format(self.modality.name)
            history_section = ""

        # Build user message (dynamic task-specific content)
        knowledge_section = ""
        if domain_knowledge:
            knowledge_section = KnowledgeElicitor.format_knowledge_context(
                domain_knowledge) + "\n\n"

        user_prompt = f"""{knowledge_section}{history_section}Task: {task}

{output_format}"""

        # Use system + user messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.model.invoke(messages)
        content = response.content
        return content if isinstance(content, str) else str(content)


class VisualAgent:
    """VISUAL modality agent for evidence presentation."""

    def __init__(self, agent_id: str, model: ChatOpenAI):
        self.agent_id = agent_id
        self.model = model
        self.modality = ModalityType.VISUAL

        # Build system prompt (static role/behavioral guidance)
        self.system_prompt = VISUAL_AGENT_SYSTEM.format(
            modality_profile=MODALITY_PROFILES[ModalityType.VISUAL],
            temporal_basics=TEMPORAL_BASICS,
            evidence_rules=EVIDENCE_RULES,
        )

    def respond(
        self,
        task: str,
        time_chart_b64: Optional[str],
        freq_chart_b64: Optional[str],
        debate_history: str = "",
        round_num: int = 1,
        max_rounds: int = 1,
        other_modalities: Optional[List[str]] = None,
        domain_knowledge: str = "",
    ) -> str:
        """Generate evidence presentation."""
        # Use refinement format for round 2+
        if round_num > 1 and debate_history:
            output_format = get_refinement_format(self.modality.name)
            history_section = f"""
=== PREVIOUS ROUND EVIDENCE ===
{debate_history}
=== END PREVIOUS EVIDENCE ===
"""
        else:
            output_format = get_evidence_format(self.modality.name)
            history_section = ""

        # Build user message (dynamic task-specific content)
        knowledge_section = ""
        if domain_knowledge:
            knowledge_section = KnowledgeElicitor.format_knowledge_context(
                domain_knowledge) + "\n\n"

        user_text = f"""{knowledge_section}{history_section}Task: {task}

{output_format}"""

        # Build multimodal user content
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text}]

        if time_chart_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{time_chart_b64}", "detail": "low"},
            })
        if freq_chart_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{freq_chart_b64}", "detail": "low"},
            })

        # Use system + user messages
        if len(user_content) == 1:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_text),
            ]
        else:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_content),  # type: ignore[arg-type]
            ]

        response = self.model.invoke(messages)
        resp_content = response.content
        return resp_content if isinstance(resp_content, str) else str(resp_content)


class NumericalAgent:
    """NUMERICAL modality agent with Lookup Function tools for evidence presentation."""

    def __init__(
        self,
        agent_id: str,
        model: ChatOpenAI,
        lookup_fn: Optional[NumericalLookupFunction] = None,
        use_frequency_features: bool = True,
    ):
        self.agent_id = agent_id
        self.model = model
        self.modality = ModalityType.NUMERICAL
        self.has_tools = lookup_fn is not None
        self.use_frequency_features = use_frequency_features
        self.lookup_fn = lookup_fn

        # Store tool configuration for later agent creation
        self.tools = create_lookup_tools(
            lookup_fn, include_frequency_features=use_frequency_features) if lookup_fn else []

        # Build tool descriptions
        if self.tools:
            tool_lines = [
                "TOOLS (call `get_info` first):",
                "- get_info() → schema, stats, detected features",
                "- get_values(start, end) → time-series values by index or timestamp",
                "- get_around(center, window) → time-series values around a point",
                "- get_features(type) → 'peak'/'valley'/'trend'/'anomaly'",
            ]
            if use_frequency_features:
                tool_lines.append(
                    "- get_frequency_features() → spectral analysis")
            if lookup_fn and lookup_fn.is_multivariate:
                tool_lines.append(
                    "- get_channel_values(ch, start, end) → specific channel")
                tool_lines.append(
                    "- get_all_channels(start, end) → all channels at once")
            if lookup_fn and lookup_fn.has_indicator:
                tool_lines.append(
                    "- get_indicator(start, end) → indicator values (e.g., MACD, Bollinger Bands) of time series")
            self.tool_section = "\n".join(tool_lines)
            self.tool_section = f"""{self.tool_section}

**TOOL USAGE RULES** (STRICT):
1. THINK FIRST: Plan what you need before calling any tool.
2. MAX 5 CALLS TOTAL: After 5 calls, you MUST provide your evidence.
3. NO REPEATED CALLS: Never call the same tool with overlapping ranges.
4. EFFICIENT STRATEGY:
   - Call get_info() ONCE to understand the data
   - Use get_features() to find key points (peaks, anomalies)
   - Use get_around() for specific locations, NOT get_values() for large ranges
5. LEVERAGE HINTS: Use location hints from other analysts' observations when available."""
        else:
            self.tool_section = "Data is provided in task description."

        # Create base agent (domain knowledge will be added dynamically in respond)
        self._create_agent("")

    def _create_agent(self, domain_knowledge: str):
        """Create or recreate agent with given domain knowledge."""
        # Include domain knowledge context if available
        knowledge_section = ""
        if domain_knowledge:
            knowledge_section = KnowledgeElicitor.format_knowledge_context(
                domain_knowledge) + "\n\n"

        # Build system prompt using template
        system_prompt = NUMERICAL_AGENT_SYSTEM.format(
            modality_profile=MODALITY_PROFILES[ModalityType.NUMERICAL],
            temporal_basics=TEMPORAL_BASICS,
            tool_section=self.tool_section,
            evidence_rules=EVIDENCE_RULES,
        )
        prompt = f"{knowledge_section}{system_prompt}"

        self.agent = create_agent(self.model, tools=self.tools, prompt=prompt)

    def respond(
        self,
        task: str,
        debate_history: str,
        round_num: int = 1,
        max_rounds: int = 1,
        other_modalities: Optional[List[str]] = None,
        domain_knowledge: str = "",
    ) -> str:
        """Generate evidence presentation."""
        # Recreate agent with domain knowledge if provided
        if domain_knowledge:
            self._create_agent(domain_knowledge)

        # Use refinement format for round 2+
        if round_num > 1 and debate_history:
            output_format = get_refinement_format(self.modality.name)
            history_section = f"""
=== PREVIOUS ROUND EVIDENCE ===
{debate_history}
=== END PREVIOUS EVIDENCE ===
"""
        else:
            output_format = get_evidence_format(self.modality.name)
            history_section = ""

        user_msg = f"""{history_section}Task: {task}

{output_format}"""

        result = self.agent.invoke(
            {"messages": [HumanMessage(content=user_msg)]}, config={"recursion_limit": 50})

        # Extract final response
        if result and "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    return content if isinstance(content, str) else str(content)
        return "Unable to generate numerical analysis."


class JudgeAgent:
    """Judge agent that evaluates evidence quality and synthesizes answers."""

    def __init__(
        self,
        agent_id: str,
        model: ChatOpenAI,
        context: Dict[str, Any],
        lookup_fn: Optional[NumericalLookupFunction] = None,
        use_code_executor: bool = True,
        use_time_chart: bool = True,
        use_freq_chart: bool = True,
        use_frequency_features: bool = True,
        domain_knowledge: str = "",
        is_synthesizer: bool = False,
        text_in_task: bool = False,
    ):
        self.agent_id = agent_id
        self.model = model
        self.domain_knowledge = domain_knowledge
        self.is_synthesizer = is_synthesizer

        # Store images for visual verification
        self.time_chart_b64 = context.get(
            "time_series_chart") if use_time_chart else None
        self.freq_chart_b64 = context.get(
            "frequency_chart") if use_freq_chart else None

        # Build tools
        tools = []
        if lookup_fn:
            tools.extend(create_lookup_tools(
                lookup_fn, include_frequency_features=use_frequency_features))
        if use_code_executor:
            tools.extend(create_code_executor_tool(context))

        # Build capabilities description
        capabilities = []
        if self.time_chart_b64 or self.freq_chart_b64:
            charts = []
            if self.time_chart_b64:
                charts.append("time series")
            if self.freq_chart_b64:
                charts.append("frequency")
            capabilities.append(
                f"VISUAL: {' + '.join(charts)} charts attached")
        if lookup_fn:
            lookup_tools = ["get_info", "get_values",
                            "get_around", "get_features"]
            if use_frequency_features:
                lookup_tools.append("get_frequency_features")
            if lookup_fn.is_multivariate:
                lookup_tools.extend(["get_channel_values", "get_all_channels"])
            if lookup_fn.has_indicator:
                lookup_tools.append("get_indicator")
            capabilities.append(f"NUMERICAL: {', '.join(lookup_tools)}")
        if use_code_executor:
            capabilities.append("CODE: execute_code(code) for calculations")

        capabilities_str = "\n".join(
            f"- {c}" for c in capabilities) if capabilities else "- No tools"

        # Track SPECIFIC verification capabilities (not conflated)
        self.has_lookup_tools = lookup_fn is not None
        self.has_code_executor = use_code_executor
        self.has_charts = bool(self.time_chart_b64 or self.freq_chart_b64)
        # Track if text is actually embedded in task description (passed by caller)
        self.has_text_in_task = text_in_task

        # Build verification instruction based on ACTUAL available capabilities
        verification_sources = []
        if self.has_lookup_tools:
            verification_sources.append(
                "numerical lookup tools (get_values, get_features)")
        if self.has_code_executor:
            verification_sources.append("code executor for calculations")
        if self.has_charts:
            verification_sources.append("attached charts")
        if self.has_text_in_task:
            verification_sources.append(
                "text context (embedded in task description)")

        if verification_sources:
            max_calls = 3 if is_synthesizer else 5
            verification_instruction = f"""

MANDATORY VERIFICATION:
Before accepting ANY claim, VERIFY against available sources:
- Available: {', '.join(verification_sources)}
- Mark claims as VERIFIED, UNVERIFIED, or CONTRADICTED
- Lower weight for UNVERIFIED claims, reject CONTRADICTED claims"""

            # Only add tool usage rules if callable tools are available
            if self.has_lookup_tools or self.has_code_executor:
                strategy_lines = []
                if self.has_lookup_tools:
                    strategy_lines.append("   - Call get_info() ONCE to understand the data")
                    strategy_lines.append("   - Use get_features() to find key points, then get_around() for specific values")
                if self.has_code_executor:
                    strategy_lines.append("   - Use execute_code() for complex calculations")
                if self.has_text_in_task:
                    strategy_lines.append("   - Leverage numerical data embedded in task description")
                
                verification_instruction += f"""

**TOOL USAGE RULES** (STRICT):
1. THINK FIRST: Identify which claims need verification before calling any tool.
2. MAX {max_calls} CALLS TOTAL: After {max_calls} calls, you MUST synthesize your answer.
3. NO REPEATED CALLS: Never call the same tool with overlapping ranges.
4. EFFICIENT STRATEGY:
{chr(10).join(strategy_lines)}"""
        else:
            verification_instruction = """

NOTE: No verification sources available. Mark all claims as UNVERIFIED."""

        # Include domain knowledge context if available
        knowledge_section = ""
        if domain_knowledge:
            knowledge_section = KnowledgeElicitor.format_knowledge_context(
                domain_knowledge) + "\n\n"

        # Build role-specific system prompt
        if is_synthesizer:
            # Synthesizer evaluates reviewer answers, not analyst evidence
            prompt = f"""{knowledge_section}You are the final decision-maker for a time-series reasoning task.
You determine the correct answer based on expert reviewers' evaluations.

YOUR JOB:
- Evaluate each reviewer's reasoning quality
- Check if reviewers followed the SUGGESTED APPROACH from DOMAIN KNOWLEDGE
- Verify disputed answers against data (if about past) or domain reasoning (if about future)
- Derive your own answer if all reviewers made the same error
- Always be skeptical of the reviewers' answers; do not blindly trust them

AVAILABLE FOR VERIFICATION:
{capabilities_str}{verification_instruction}

{TEMPORAL_AWARENESS}

{SYNTHESIZER_PROTOCOL}"""
        else:
            # Reviewer evaluates analyst evidence
            prompt = f"""{knowledge_section}You are an expert reviewer for a time-series reasoning task.
You evaluate evidence from specialist analysts and synthesize a well-reasoned answer.

YOUR JOB:
- Score each analyst's evidence based on observation quality, inference logic, and honesty
- Verify claims against original data before accepting them
- Identify conflicts between different analysts
- Adjust your answer confidence based on how well evidence agrees
- Check claims against domain knowledge (if provided above)

AVAILABLE FOR FACT-CHECKING:
{capabilities_str}{verification_instruction}

{TEMPORAL_AWARENESS}

{JUDGE_EVALUATION_CRITERIA}

{JUDGE_PROTOCOL}"""

        self.agent = create_agent(model, tools=tools, prompt=prompt)

    def respond(
        self,
        task: str,
        full_debate_history: str,
        as_final: bool = False,
        available_modalities: Optional[List[str]] = None,
    ) -> str:
        """Evaluate evidence quality and synthesize answer."""
        available_modalities = available_modalities or ["TEXT", "VISUAL", "NUMERICAL"]
        content = []

        if as_final:
            text_msg = task
        else:
            # Build score template
            score_lines = [
                f"- {mod}: (Observation: _/30, Inference: _/50, Honesty: _/20) = [0-100]" for mod in available_modalities]
            score_template = "\n".join(score_lines)

            # Build weights template
            weights_lines = [f"- {mod}: [X%]" for mod in available_modalities]
            weights_template = "\n".join(weights_lines)

            # Build evidence template
            # evidence_lines = [
                # f"- {mod} ([X%]): [key observations/inferences used]" for mod in available_modalities]
            # evidence_template = "\n".join(evidence_lines)

            # Build verification sources list based on ACTUAL capabilities
            verification_sources = []
            if self.has_lookup_tools:
                verification_sources.append("lookup tools")
            if self.has_code_executor:
                verification_sources.append("code executor")
            if self.has_charts:
                verification_sources.append("charts")
            if self.has_text_in_task:
                verification_sources.append("text in task")

            # Add domain knowledge as a verification source if available
            if self.domain_knowledge:
                verification_sources.append("domain knowledge (above)")

            # Build unified verification section (data + domain in one)
            if verification_sources:
                verification_str = ", ".join(verification_sources)
                verification_section = f"""VERIFICATION (check against {verification_str}):
- [Claim]: [VERIFIED/UNVERIFIED/CONTRADICTED] + [DOMAIN: MATCHES/VIOLATES/N-A] - [explanation]
- [Claim]: [VERIFIED/UNVERIFIED/CONTRADICTED] + [DOMAIN: MATCHES/VIOLATES/N-A] - [explanation]
(For each major claim: first check domain knowledge, then check data sources)"""
            else:
                # No verification sources at all
                verification_section = """VERIFICATION:
- No verification sources available
- Mark all claims as UNVERIFIED"""

            text_msg = f"""Task: {task}

Evidence:
{full_debate_history}

ANALYSTS: {', '.join(available_modalities)}

YOUR JOB: Score evidence, VERIFY claims (data + domain), detect CONFLICTS, synthesize CALIBRATED answer.

OUTPUT FORMAT (keep everything concise):
TASK: <Restate what the task is asking in one sentence>
TASK TYPE: [FUTURE / PAST-PRESENT]

SCORES: {score_template}
WEIGHTS: {weights_template}

{verification_section}
**IMPORTANT**: Never verify claims against data for FUTURE tasks. Past time series CANNOT verify future predictions. Always check the requested date against the date at which the data actually ends.
Do NOT refer to past data calculation or verification to support the answer. Only past data vs. future data (expected values by YOUR prediction) can be used to support the answer.

OUTSTANDING CONFLICTS: [NO_CONFLICT / DETECTED / RESOLVED] - <details>
KEY EVIDENCE: <main observations/inferences used>
CALIBRATED ANSWER: [answer in exact task required format]"""

        content.append({"type": "text", "text": text_msg})

        # Attach charts for fact-checking
        if self.time_chart_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self.time_chart_b64}", "detail": "low"},
            })
        if self.freq_chart_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self.freq_chart_b64}", "detail": "low"},
            })

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                if len(content) > 1:
                    result = self.agent.invoke(
                        {"messages": [HumanMessage(content=content)]}, config={"recursion_limit": 50})
                else:
                    result = self.agent.invoke(
                        {"messages": [HumanMessage(content=text_msg)]}, config={"recursion_limit": 50})

                if result and "messages" in result:
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage) and msg.content:
                            resp_content = msg.content
                            return resp_content if isinstance(resp_content, str) else str(resp_content)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    continue  # Retry on next attempt
                # Log final failure but don't crash
                break

        return "SCORES: [Failed]\nSYNTHESIS_WEIGHTS: [Failed]\nANSWER: ABSTAIN"
