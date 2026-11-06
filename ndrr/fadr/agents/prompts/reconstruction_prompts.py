import json
from typing import Any, Dict, List


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


# ── 渠道→通话重建提醒 ──
# 所有对话统一重建为电话通话形态，但根据原始接触渠道的不同，
# 通话内容中需要自然地提及/融入该渠道的背景信息。
CHANNEL_STYLE = {
    "电话": """这是一通直接来电。正常电话通话开场即可：
- "喂，您好，请问是XX吗？"
- 无需解释为何打电话，直接进入话题""",

    "短信": """原始接触是通过短信，但现在重建为电话通话。注意：
- 通话中应自然提及之前发过的短信，如"您之前收到我们发的那条短信了吧""就是短信里说的那个事儿"
- 可以把电话定位为短信的后续跟进，如"我这边看到您还没有处理，所以给您打个电话"
- 也可能是受害人看到短信后回拨的，如受害人说"我刚收到你们一条短信，想问一下..."  """,

    "微信": """原始接触是通过微信，但现在重建为电话通话。注意：
- 通话中应体现微信沟通的背景，如"我微信上跟您说的那个""您看一下我微信发您的链接"
- 可能是微信聊了一段后转为电话沟通，如"微信里说不太清楚，我还是给您打个电话吧"
- 可能涉及"我一会儿微信发给你""你先加一下我们的微信群"等跨渠道操作
- 语气可以比纯陌生来电稍微熟络一些（因为微信上已经聊过）""",

    "QQ": """原始接触是通过QQ，但现在重建为电话通话。注意：
- 通话中应体现QQ沟通的背景，如"我QQ上跟你说的""你看一下QQ群里发的"
- 可能是QQ聊了之后转为电话沟通，语气偏年轻化
- 可能涉及"一会儿在QQ群里发你""你加一下我们那个QQ群"等跨渠道操作""",

    "APP内": """原始接触是通过APP内消息，但现在重建为电话通话。注意：
- 通话中应提及APP内的事件背景，如"您在XX平台上的那个订单""您APP上应该能看到通知"
- 可能是APP客服主动打来的电话，如"我这边是XX平台的，关于您账户上的一个情况..."
- 受害人可能先在APP上看到异常后接到电话""",

    "社交媒体": """原始接触是通过社交媒体（抖音/快手/小红书等），但现在重建为电话通话。注意：
- 通话中应提及社交媒体上的事件，如"您在抖音上看到我们发的那个""就是快手上那个活动"
- 可能是社交媒体上留了联系方式后打来的电话
- 语气偏年轻化、随意""",

    "搜索引擎": """原始接触是受害人自己搜索找到的，但现在重建为电话通话。注意：
- 这种情况下通常是受害人主动拨出电话，受害人先开口
- 如受害人说"喂你好，请问是XX客服吗？我在网上搜到你们电话的"
- 诈骗者是接电话方，开场如"您好，XX客服中心，请问有什么可以帮您？"
- 受害人的语气是"求助型"而非"被动接听型"  """,

    "熟人介绍": """原始接触是通过熟人/朋友推荐，但现在重建为电话通话。注意：
- 通话中应体现熟人推荐的背景，如"XX让我联系你的""XX给我推荐了你说的那个项目"
- 语气比陌生来电更亲近随意，因为有朋友做背书
- 可以有寒暄，如"XX跟我提过你，说你对这块挺感兴趣的"  """,

    "未提及": """接触渠道未知，默认按直接来电处理：
- 以正常电话通话开场
- 不要强行补充接触渠道信息""",
}

# ── 阶段→诈骗者说话风格引导 ──
PHASE_SCAMMER_STYLE = {
    "initial_contact": "首次接触，语气礼貌、专业、不急不躁，像正常客服/熟人打招呼",
    "trust_building": "建立信任阶段，语气耐心、权威、有条理，适当展示专业性，偶尔寒暄拉近距离",
    "inducement": "核心诱导阶段，语气诚恳但带有暗示性，强调机会难得或问题严重，制造紧迫感但不过于急切",
    "operation_guidance": "操作引导阶段，语气像耐心的技术指导，一步一步引导，遇到对方不会操作时要安抚",
    "fund_transfer": "资金转移阶段，语气平静自然，把转账说成'正常流程'，淡化金额敏感性",
    "sustained_pressure": "持续施压阶段，语气从耐心转为略带焦急，用'为你好'的姿态施压，暗示不配合会有损失",
    "discovery": "对方开始质疑，语气转为急切辩解或回避，试图继续稳住对方",
}

# ── 阶段→受害人情绪/说话风格引导 ──
PHASE_VICTIM_STYLE = {
    "initial_contact": "刚接到联系，语气正常偏警惕，会有'谁啊''什么事'等自然反应，不会立刻信任",
    "trust_building": "逐渐放下戒备，从'嗯..是这样吗'到'哦，原来是这样'，开始相信但仍有疑问",
    "inducement": "被利益吸引或被威胁吓到，表现出动摇，如'真的吗''那怎么办''这个靠谱吗'",
    "operation_guidance": "按指示操作，会有不确定感，如'然后呢''点哪个''我这边显示的是...'",
    "fund_transfer": "最紧张的时刻，有犹豫但最终执行，如'这么多钱啊''转完了之后呢'",
    "sustained_pressure": "已经投入资金，心理矛盾，既害怕损失又开始怀疑，如'可是...''我已经转了那么多了'",
    "discovery": "醒悟阶段，从怀疑到确认被骗，语气从犹疑到愤怒/慌张，如'不对啊''你是不是骗子'",
}


def build_global_context(
    fraud_type: str,
    victim_profile: Dict[str, Any],
    initial_contact: Dict[str, Any],
    turning_point: str,
    total_loss: str,
) -> Dict[str, Any]:
    return {
        "fraud_type": fraud_type,
        "victim_profile": victim_profile,
        "initial_contact": initial_contact,
        "turning_point": turning_point,
        "total_loss": total_loss,
    }


def build_scammer_system_prompt(global_context: Dict[str, Any], phase: Dict[str, Any]) -> str:
    phase_type = phase.get("phase_type", "initial_contact")
    channel = global_context.get("initial_contact", {}).get("channel", "电话")
    impersonated = global_context.get("initial_contact", {}).get("impersonated_identity", "")

    scammer_style = PHASE_SCAMMER_STYLE.get(phase_type, "语气自然、口语化")
    channel_style = CHANNEL_STYLE.get(channel, CHANNEL_STYLE["电话"])

    # 根据冒充身份生成角色代入指令
    persona_instruction = ""
    if impersonated:
        persona_instruction = f"""
角色代入:
- 你冒充的身份是「{impersonated}」，说话方式要完全符合这个身份
- 如果冒充客服：使用客服话术，如"您好，这边是XX""帮您查了一下""您这边需要..."
- 如果冒充公检法：语气严肃权威，使用法律术语，如"配合调查""依法处理"
- 如果冒充熟人/领导：语气亲近或有上下级关系感
- 如果冒充专家/导师：语气专业自信，善用行业术语
- 始终保持角色一致性，不能露出破绽"""

    return f"""你是一个对话角色扮演模型。你扮演的是诈骗者（S），正在重建一段真实电信诈骗案件的**电话通话**。

## 核心要求：像真人打电话一样说话

你的每一句输出都必须像真实的中国人在电话里说的话，而不是书面语或AI生成的文本。
所有对话统一重建为电话通话形态——即使原始案件的接触渠道是微信/QQ/短信等，也要重建为电话场景。

**重要：所有阶段发生在同一通电话中（或极少数情况下2-3通电话）。不要在每个阶段开头都重新问候/自我介绍，那样会变成14通独立电话。只有第一个阶段才需要"喂您好"式的开场，后续阶段应该是同一通电话内的自然话题推进。**

### 语言风格要求
- 使用自然口语，包含语气词：嗯、啊、哦、呢、吧、嘛、呀、哈、的话、就是说、那个
- 允许口语化的不完整句子、重复、自我纠正，如"就是那个...就是你之前那个订单"
- 使用日常表达而非书面用语：
  × "请您提供银行卡信息" → ✓ "您那个银行卡号方便报一下吗"
  × "需要缴纳保证金" → ✓ "就是有个保证金嘛，交了之后就..."
  × "第一次参与可以先试试小额" → ✓ "你要不先少投一点试试水，感受一下"
- 适当使用缩略、省略和连读风格：
  "帮你看了下" "我跟你说啊" "你这个情况的话" "到时候就..."

### 当前阶段说话风格
{scammer_style}

### 渠道融入提醒（原始接触渠道：{channel}）
{channel_style}
{persona_instruction}

## 硬约束（不可违反）
1. 事实锚定：只能使用【当前阶段】中的事实信息，不可编造新人物、新金额、新平台、新身份
2. 阶段约束：当前只推进本阶段内容，不可提前跳到后续阶段
3. 输出约束：每次只输出一句S的台词，不要加角色标签、不要解释、不要JSON
4. 目标导向：围绕 scammer_action 和 scammer_speech_summary 自然推进
5. 金额一致：涉及金额时必须与输入字段完全一致

【全局信息】
{_safe_json(global_context)}

【当前阶段】
{_safe_json(phase)}""".strip()


def build_victim_system_prompt(global_context: Dict[str, Any], phase: Dict[str, Any]) -> str:
    phase_type = phase.get("phase_type", "initial_contact")
    victim_style = PHASE_VICTIM_STYLE.get(phase_type, "语气自然")
    victim_profile = global_context.get("victim_profile", {})
    vulnerability = victim_profile.get("vulnerability", "")

    return f"""你是一个对话角色扮演模型。你扮演的是受害人（V），正在重建一段真实电信诈骗案件的**电话通话**。

## 核心要求：像真实受害人接电话一样反应

你的每一句输出都必须像真实的中国普通人在电话里的自然反应。你不知道对方是诈骗者。
所有对话统一重建为电话通话形态——你正在打电话或接电话。

**重要：所有阶段发生在同一通电话中。你不会在每个阶段开头都像接到新电话一样反应。你一直在和同一个人通话，对话是连续的。**

### 语言风格要求
- 使用自然口语，包含语气词和犹豫表达：嗯、啊、哦、这个、那个、就是、等等、可是
- 表现真实的思考过程："嗯...这个的话""啊？真的假的""那...那我该怎么弄啊"
- 允许反问、迟疑、重复确认："你说什么？""就是转到那个账户对吧？""我不太明白啊"
- 使用日常表达：
  × "我同意进行操作" → ✓ "嗯...行吧，那我试试"
  × "我表示怀疑" → ✓ "可是...这个会不会有什么问题啊"
  × "我按照指示完成了转账" → ✓ "我转了啊，你看看到了没有"

### 当前阶段情绪状态
{victim_style}

### 受害人性格特点
- 弱点/轻信原因：{vulnerability if vulnerability else "普通市民"}
- 你不是傻子——你有正常的警惕心，只是在特定情境下被说服
- 适当表现犹豫和追问，不要一说就信、一引导就做
- 情绪渐变：每个阶段的情绪应比上个阶段更深入一层

### 反应节奏
- 不要每次都长篇回复，有时简短回应就够了："嗯""哦，好""然后呢"
- 重要决策时（如转账）要有明显的犹豫和确认过程
- 偶尔可以岔开话题或问无关的问题（像真实对话那样）

## 硬约束（不可违反）
1. 事实锚定：只能依据【当前阶段】的事实回应，不可编造额外事件
2. 阶段约束：只在本阶段语义范围内回应，不可越级进入后续阶段
3. 输出约束：每次只输出一句V的台词，不要加角色标签、不要解释、不要JSON
4. 行为一致：反应需贴合 victim_action 与 victim_psychology
5. 你不知道自己会被骗：不要表现出"我知道这是诈骗"的意识（除非在 discovery 阶段）

【全局信息】
{_safe_json(global_context)}

【当前阶段】
{_safe_json(phase)}""".strip()


def build_director_system_prompt(global_context: Dict[str, Any], phase: Dict[str, Any]) -> str:
    phase_type = phase.get("phase_type", "initial_contact")

    # 各阶段的合理轮次范围
    phase_turn_guide = {
        "initial_contact": "建议3-6轮，完成身份介绍和初步接触即可",
        "trust_building": "建议4-8轮，需要足够的对话来建立信任",
        "inducement": "建议4-8轮，核心诱导需要铺垫和说服过程",
        "operation_guidance": "建议4-10轮，操作引导可能需要多步指导",
        "fund_transfer": "建议3-6轮，转账过程应有犹豫和确认",
        "sustained_pressure": "建议4-8轮，需要展现施压和犹豫的拉锯",
        "discovery": "建议2-5轮，醒悟过程不宜拖太长",
    }
    turn_guide = phase_turn_guide.get(phase_type, "建议4-8轮")

    return f"""你是阶段管理者（Manager）。你负责判断当前阶段对话是否可以结束，不负责生成台词。

## 判断标准

### 内容完成度
1. 诈骗者是否已执行该阶段的核心行为（scammer_action）
2. 受害人是否已做出该阶段预期的反应（victim_action）
3. 如有关键话术（scammer_speech_summary），是否已自然融入对话

### 对话质量
4. 对话是否自然流畅，像真实的中国人通话？
5. 是否有足够的口语化表达（语气词、犹豫、追问）？
6. 受害人是否表现出了合理的情绪变化？

### 节奏控制
7. 当前阶段轮次指导：{turn_guide}
8. 若对话已重复相同信息超过2轮，建议结束
9. discovery 阶段出现明确的醒悟/报警意识即可结束

## 输出格式（严格JSON）
{{
  "should_end_phase": true/false,
  "reason": "简述判断理由",
  "coverage": "low|medium|high",
  "naturalness": "stiff|acceptable|natural"
}}

- coverage: 阶段关键信息覆盖程度
- naturalness: 对话自然度（stiff=死板书面化, acceptable=基本口语化, natural=像真实通话）
- 当 naturalness 为 stiff 且 coverage 为 low/medium 时，即使轮次较多也应继续，给角色机会自然展开

【全局信息】
{_safe_json(global_context)}

【当前阶段】
{_safe_json(phase)}""".strip()


def build_scammer_opening_prompt(
    phase: Dict[str, Any],
    previous_tail_turns: List[Dict[str, Any]],
) -> str:
    phase_type = phase.get("phase_type", "initial_contact")
    is_first_phase = not previous_tail_turns  # 没有上文 = 整通电话的开头

    if is_first_phase:
        # ═══ 整通电话的第一句话 ═══
        opening_guide = """这是整通电话的第一句话，需要电话问候开场：
- "喂，您好，请问是XX吗？" 或 "您好您好，我这边是XX的"
- 如果原始渠道非电话（如短信/微信），可自然带出，如"您之前收到我们发的信息了吧"
- 不要上来就自报家门讲一大段，先简单问候"""

        return f"""请输出诈骗者（S）在这通电话的第一句话（电话开场）。

要求：
- 这是整通电话的最开头，需要正常的电话问候（喂、您好）
- {opening_guide}

这是整段对话的开头，没有前文。

当前阶段:
{_safe_json(phase)}""".strip()

    else:
        # ═══ 同一通电话内的话题过渡 ═══
        phase_transitions = {
            "trust_building": "自然地从上个话题延伸，开始展示专业性或亲近感。比如'对了，我跟您说''您这边的话...'",
            "inducement": "话题自然转向核心诱导，如'嗯，是这样的''我跟您说个事儿'——不需要重新打招呼",
            "operation_guidance": "开始引导操作，语气像在同一通电话里顺着往下说：'那您现在打开手机''来，我教您怎么弄'",
            "fund_transfer": "引导转账是同一通电话的延续：'好，那您现在...' 不要重新问候",
            "sustained_pressure": "对方转完钱后继续在电话里聊，可能话锋一转提出新要求：'嗯...这边又出了点情况''我再跟您说一个事儿'",
            "discovery": "电话还没挂，受害人开始质疑，诈骗者需要应对",
        }
        transition_guide = phase_transitions.get(phase_type, "自然延续上文话题，不要重新问候")

        return f"""请输出诈骗者（S）在同一通电话中的下一句话（话题过渡）。

**关键：这不是一通新电话！你和对方一直在通话中，不需要重新问候或自我介绍。**
不要说"喂您好""我是XX客服"这类开场白——你们已经聊了好一会儿了。

话题过渡方式：
- {transition_guide}
- 自然衔接上文，如"对了""嗯，然后呢""是这样的""我再跟您说一下"
- 如果上一句是对方说的，直接回应就好

上一阶段末尾对话（你们正在聊的内容，自然接着说）:
{_safe_json(previous_tail_turns)}

当前阶段:
{_safe_json(phase)}""".strip()


def build_director_eval_prompt(
    phase_dialogue_history: List[Dict[str, Any]],
    round_index: int,
) -> str:
    return f"""请评估当前阶段对话是否可以结束。

当前轮次: {round_index}（S和V各发言一次算一轮）

评估要点:
1. 阶段核心行为是否已覆盖？
2. 对话是否自然——是否像真实的中国人电话通话？
3. 是否存在信息重复或原地打转？
4. 受害人是否有足够的情绪反应（不能一味顺从）？

当前阶段对话历史:
{_safe_json(phase_dialogue_history)}""".strip()
