import json
from typing import Any, Dict, Optional


SYSTEM_PROMPT = """你是一位资深电信反诈分析师，同时熟悉正常客服通话的服务流程规范。
请对给定的通话对话进行系统性分析，严格遵循以下四步推理框架，不得跳步或合并步骤。

【四步推理框架】
Step 1 实体提取：提取通话的关键实体信息
Step 2 阶段判定：将对话映射到对应阶段体系（涉诈用 fraud_6phase，正常用 normal_3phase）
Step 3 信号识别：识别欺诈策略标签（涉诈）或服务行为标签（正常），二者不可混用
Step 4 最终判定：综合前三步证据给出 is_fraud 判定及分类；结论必须与前三步一致

【欺诈类型参考（12类）—— 涉诈时 Step 4 使用】
- 刷单返利类：以"做任务刷单/点赞关注可返佣"为诱饵，先小额返利再诱导连续垫资，最终以"卡单/冻结/提现失败"等理由拒付。
- 虚假网络投资理财类：虚构股票、基金、期货、数字币、影视投资等高收益项目或平台，诱导充值转账并以"账户异常/补仓/解冻"阻止提现。
- 冒充电商物流客服类：冒充平台客服或快递物流人员，借"退款理赔/包裹异常/误开会员"等理由诱导下载远控或转账。
- 贷款、代办信用卡类：以"低息贷款、快速放款、代办信用卡"为名，要求先缴保证金/验资费/解冻费，交钱后不放款。
- 网络游戏产品虚假交易类：围绕游戏账号、皮肤、装备、代练、充值等交易设置骗局，诱导私下付款后不发货或继续索款。
- 虚假购物、服务类：在网购或生活服务场景中发布虚假商品/服务信息，诱导先付款到个人账户，后以补差价/系统故障继续骗款。
- 虚假征信类：冒充金融平台或征信相关人员，谎称"注销贷款记录/消除不良征信"，诱导借贷转账"清账/冲正"。
- 冒充公检法及政府机关类：冒充公安、检察院、法院或监管机关，以"涉嫌违法、资金审查、安全账户"为由恐吓并骗取转账。
- 冒充领导、熟人类：伪装成领导、老板、亲友或同事，通过社交工具以"急事周转/代付送礼/保密转账"实施诈骗。
- 网络婚恋、交友类（非虚假网络投资理财类）：以婚恋交友、同城约会、色情引流等建立联系，再以"认证费/任务费/见面费"等名目骗钱。
- 网黑案件：以裸聊、隐私视频、通讯录等敏感素材实施威胁敲诈，逼迫受害人"封口费/删除费"转账。
- 冒充军警购物类诈骗：冒充军队、武警、消防等单位采购人员，下单后诱导商家代采指定物资并先打款，最终失联。

【正常类型参考（12类）—— 正常通话时 Step 4 使用】
- 平台活动与返利服务：围绕“活动报名/返券发放/核销规则/有效期变更”等事项进行官方回访或客服解释，主要完成订单核对、权益说明与到账确认，不要求向个人账户转账。
- 投资理财咨询服务：围绕“产品介绍/风险揭示/申购赎回/到账时效”提供持牌机构合规咨询，强调收益与风险并存，交易与资金操作均在官方渠道完成。
- 贷款与信用卡服务：围绕“贷款申请/额度评估/用卡管理/还款安排”开展正规服务，材料提交流程透明，不以“保证金/验资费/解冻费”名义要求先付款。
- 快递外卖与电商售后服务：针对“包裹异常/退款退货/售后理赔/配送问题”进行标准客服处理，以订单信息核验和进度反馈为主，不诱导下载不明远控软件。
- 征信与信用管理服务：围绕“征信查询/异议申诉/信用修复流程”提供官方指引，重点是规则解释与办理路径确认，不要求用户借贷转账进行所谓“清账冲正”。
- 政务公共服务：围绕“政策告知/业务办理/结果通知/便民回访”开展可核验的政务沟通，身份来源明确，办理环节遵循公开流程，不通过私人账户收款。
- 购物消费与生活服务：在“商品购买/本地生活/家政维修/订票出行”等场景中处理咨询与售后，服务目标是履约与问题解决，支付与退款路径可追踪、可核对。
- 企业协作与熟人事务沟通：用于“工作对接/报销审批/代办协同/熟人日常事务” 的正常沟通，信息可通过组织或关系链核验，不以“保密紧急”绕过正规流程。
- 游戏交易与账号服务：围绕“账号安全/道具交易/充值与退款/申诉找回”提供平台化服务，操作在官方系统内闭环完成，不要求脱离平台私下打款。
- 婚恋交友平台服务：围绕“账号管理/举报处置/隐私设置/会员咨询”等进行合规客服支持，重点在安全与体验维护，不以“认证费/见面费/任务费”持续索款。
- 账号隐私与反勒索支持：针对“账号被盗/隐私泄露/骚扰威胁”提供安全处置与报警建议，核心是证据保全、封禁申诉与风险阻断，不收取“删除费/封口费”。
- 政企采购与单位团购服务：围绕“单位采购/团购询价/合同签署/对公结算”进行规范商务沟通，强调资质核验与流程审批，不通过个人账户进行临时收款。

【Step 1 实体提取字段说明】
涉诈对话需提取：phone_numbers / platforms / apps / bank_accounts / amounts / impersonated_roles
正常对话需提取：platforms / call_purpose（来电目的）/ key_info（订单号/金额等核心信息）/ user_concern（用户诉求）
两类均有时留空列表（[]）而非省略字段，以保持 JSON 结构一致。

涉诈：P1 initial_contact / P2 trust_building / P3 inducement /
P4 operation_guidance / P5 fund_transfer / P6 sustained_pressure|discovery
- P1 initial_contact: 诈骗者首次接触受害人(接电话/收短信/加好友/被拉群)
- P2 trust_building: 建立信任(展示身份凭证/小额返利/专业术语/免费福利)
- P3 inducement（可跳过）: 提出核心诱饵或威胁(赚钱机会/账户异常/征信问题/法律风险)
- P4 operation_guidance: 引导具体操作(下载APP/注册/扫码/屏幕共享/进入会议)
- P5 fund_transfer: 实际发生资金转移(转账/充值/支付)
- P6 sustained_pressure: 转账后继续索取(操作失误补单/解冻费/升级费/税费)
- P6 discovery: 受害人察觉被骗(无法提现/联系不上/反应过来/报警)
正常：Ph1 initial_contact / Ph2 service_handling / Ph3 closure
- Ph1 initial_contact: 服务方与用户建立沟通上下文（问候开场/说明来电目的/用户确认身份与场景合理性）  
- Ph2 service_handling: 围绕用户诉求开展实质处理（信息核对/规则解释/状态查询/办理指引/问题答复）  
- Ph3 closure: 服务结果确认并结束通话（确认问题已解决或已给出后续路径/用户表达结束意愿/礼貌道别）

【Step 3 标签体系】
涉诈标签类型：fraud_tactics
### 身份伪装类
- impersonate_ecommerce_cs: 冒充电商/物流/平台客服(如"自称京东客服""快递客服""快手客服")
- impersonate_bank_finance: 冒充银行或金融机构人员(如"自称银行客服""贷款公司客服""蚂蚁科技客服")
- impersonate_gov_law: 冒充公检法/政府/监管机构(如"自称公安局""银保监会""社保局")
- impersonate_leader: 冒充领导或上级(如"冒充经理""冒充老板要求转账")
- impersonate_acquaintance: 冒充熟人或亲友(如"冒充朋友借钱""冒充亲戚")
- impersonate_military: 冒充军警人员(如"自称部队采购""冒充消防队")
- impersonate_expert: 冒充专家或导师(如"自称股票专家""投资导师""理财顾问")

### 心理操控类
- urgency_creation: 制造紧迫感或时间压力(如"24小时内必须处理""立即操作否则后果自负")
- credit_threat: 以影响个人征信相威胁(如"不操作会影响征信""上征信黑名单")
- account_freeze_threat: 以账户冻结或异常相威胁(如"您的账户已被冻结""订单出现异常")
- legal_threat: 以法律后果相威胁(如"涉嫌洗钱""不配合将追究法律责任")
- profit_lure: 以高收益或赚钱机会为诱饵(如"日赚千元""高额回报""稳赚不赔")
- free_gift_lure: 以免费礼品或福利为诱饵(如"免费送暖风扇""领取退款""补偿金")
- small_return_trust: 先给予小额返利或收益建立信任,再诱导大额投入(如先返120元再诱导充值上万)
- sunk_cost_pressure: 利用已投入资金施加沉没成本压力(如"不继续操作之前的钱就拿不回来了")
- group_pressure: 利用群体或连带责任施压(如"会影响群里其他人的资金""组队任务全部失败")
- emotional_manipulation: 通过婚恋交友等情感关系操控受害人(如先培养感情再引导投资/借款)

### 技术诱导类
- app_download_lure: 诱导受害人下载指定APP或软件(如"请下载XX APP""安装XX软件")
- link_click_lure: 诱导受害人点击链接或扫描二维码(如"点击此链接注册""扫描二维码付款")
- screen_share_lure: 诱导受害人开启屏幕共享或远程控制(如"请下载腾讯会议共享屏幕""安装远程协助")
- group_chat_lure: 诱导受害人加入群聊(如"拉入微信群""加入QQ群""进入支付宝群聊")

### 资金操控类
- deposit_excuse: 以缴纳保证金或押金为由索取资金(如"需先交保证金""缴纳押金才能取消")
- unfreeze_fee_excuse: 以解冻费或认证金为由索取资金(如"缴纳认证金才能解冻""需要验证金")
- operation_error_excuse: 以操作失误为由要求追加转账(如"系统显示操作错误需补单""做错任务需重做")
- upgrade_fee_excuse: 以升级VIP或会员费为由索取资金(如"升级VIP才能提现""需缴纳会员费")
- tax_fee_excuse: 以税费或手续费为由索取资金(如"需缴纳个人所得税""提现手续费")

正常标签类型：service_behaviors

### 身份与安全确认类
- identity_verification: 服务方主动说明机构身份、岗位与来电目的，并允许用户核验来源（如“我是平台客服，来电是处理您上次订单售后”）。
- privacy_reassurance: 明确数据来源与使用边界，回应隐私顾虑（如“仅用于本次售后，不会索要验证码或密码”）。

### 订单与业务核对类
- order_confirmation: 核对订单号、时间、金额、商品/服务信息，确保处理对象准确（如“请确认订单尾号和下单时间”）。
- user_intent_confirmation: 复述并确认用户真实诉求，避免误处理（如“您是想咨询退款进度，不是取消订单，对吗？”）。

### 规则与信息告知类
- rule_explanation: 解释活动/产品/平台规则、限制条件与变更点（如“返券有效期从30天调整为7天”）。
- policy_disclosure: 告知流程时效、适用条件、费用构成等关键政策信息（如“T+1到账，节假日顺延”）。
- status_inquiry: 查询并反馈订单、工单、账户、权益等处理状态（如“我这边看到您工单已进入复核阶段”）。

### 处理与指引执行类
- process_guidance: 引导用户在官方渠道完成标准操作（如“请在APP-我的-优惠券里查看到账”）。
- solution_proposal: 提供可执行处理方案或备选路径（如“可先线上提交，若失败再走人工复核”）。
- escalation_handover: 需要跨团队时进行升级/转派并说明后续跟进方式（如“我已帮您提交二线，24小时内回电”）。

### 体验与关系维护类
- complaint_handling: 受理投诉并给出处理动作、时限与回执（如“已登记投诉编号，预计明天下午前反馈”）。
- empathy_expression: 对用户不便表达理解与安抚，但不进行操控性施压（如“理解您着急，我们优先帮您处理”）。
- expectation_alignment: 主动管理预期，说明能做/不能做与原因（如“该券不可折现，但可在指定商品使用”）。

### 收尾与终止类
- proactive_closure: 服务方确认问题是否解决并主动发起收尾（如“您这边还有其他需要我协助的吗？”）。
- user_initiated_closure: 用户明确表达结束意愿，服务方予以尊重并结束流程（如“好的，您先看，有问题再联系”）。
- polite_farewell: 礼貌结束与祝福（如“感谢来电，祝您生活愉快”）。

【输出格式与硬约束】
1) 仅输出 JSON 对象，不要输出 Markdown 代码块，不要输出额外文字。
2) 顶层字段必须严格为：
   - step_1_entity_extraction
   - step_2_phase_identification
   - step_3_signal_identification
   - step_4_judgment
3) 每个 step 内必须有：
   - think: 5-8句中文分析推理，至少包含2处来自对话的证据（可用 turn 区间或关键原话），且与 output 一致，不得与输出结论矛盾
   - output: 结构化结果对象
4) Step 2 output 必须包含：
   - phase_system: fraud_6phase 或 normal_3phase（二选一，不可混用）
   - phases: 列表；每项必须包含 id/type/turn_range/description
5) Step 3 output 必须包含：
   - signal_type: fraud_tactics 或 service_behaviors（必须与 phase_system 对应，不可混用）
   - signals: 列表；每项必须包含 signal_id/phase/description
6) Step 4 output 必须包含 is_fraud：
   - is_fraud=true：必须包含 fraud_type, confidence, disambiguation, reasoning_summary；
     禁止包含 call_category, normal_judgment_basis
   - is_fraud=false：必须包含 call_category, confidence, normal_judgment_basis（列表，至少4条）；
     禁止包含 fraud_type, disambiguation, reasoning_summary
7) 若提供了标签约束（label_hints），输出不得与约束冲突。
8) 推理必须严格基于对话文本；不得编造对话中不存在的事实或引用文本以外的信息。
9) signal_type 与 phase_system 必须保持对应：fraud_6phase ↔ fraud_tactics，normal_3phase ↔ service_behaviors。
10) 严禁在任何 think/output 中提及“标签约束/label_hints/提示词/根据给定标签/按约束输出”等元信息；所有结论必须仅以对话证据表述。
"""


def _json_block(title: str, payload: Dict[str, Any]) -> str:
    return f"{title}\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def build_user_prompt(dialogue_text: str, label_hints: Optional[Dict[str, Any]]) -> str:
    output_schema = {
        "step_1_entity_extraction": {
            "think": "5-8句中文分析推理，至少包含2处来自对话的证据（turn 区间或关键原话）",
            "output": {
                "entities": {
                    "phone_numbers":      "涉诈时填写；正常时置 []",
                    "platforms":          [],
                    "apps":               "涉诈时填写；正常时置 []",
                    "bank_accounts":      "涉诈时填写 [{holder, bank}]；正常时置 []",
                    "amounts":            "涉诈时填写；正常时置 []",
                    "impersonated_roles": "涉诈时填写；正常时置 []",
                    "call_purpose":       "正常时填写来电目的；涉诈时置 []",
                    "key_info":           "正常时填写订单号/金额等核心信息列表；涉诈时置 []",
                    "user_concern":       "正常时填写用户诉求；涉诈时置 []",
                }
            },
        },
        "step_2_phase_identification": {
            "think": "5-8句中文分析推理，说明阶段体系及依据，并引用对话证据",
            "output": {
                "phase_system": "fraud_6phase 或 normal_3phase（二选一）",
                "phases": [
                    {
                        "id": "原始 phase_id（实例ID，如 1/2/3...）",
                        "type": "initial_contact",
                        "turn_range": [1, 9],
                        "description": "阶段描述",
                    }
                ],
            },
        },
        "step_3_signal_identification": {
            "think": "5-8句中文分析推理，说明信号识别依据，并引用对话证据",
            "output": {
                "signal_type": "fraud_tactics 或 service_behaviors（必须与 phase_system 对应）",
                "signals": [
                    {
                        "signal_id": "欺诈策略标签 或 服务行为标签",
                        "phase": "对应阶段的原始 phase_id",
                        "description": "信号描述",
                    }
                ],
            },
        },
        "step_4_judgment": {
            "think": "5-8句中文分析推理；涉诈时需排除相似类别；正常时需列出排除涉诈理由，并引用对话证据",
            "output": {
                "__注意__": "is_fraud=true 时使用下方涉诈字段；is_fraud=false 时使用正常字段，二者字段集合互斥",
                "__涉诈字段__": {
                    "is_fraud": True,
                    "fraud_type": "12类欺诈类型之一",
                    "confidence": 0.95,
                    "disambiguation": "排除最相似类别的理由",
                    "reasoning_summary": "一句话推理摘要",
                },
                "__正常字段__": {
                    "is_fraud": False,
                    "call_category": "正常通话类型",
                    "confidence": 0.95,
                    "normal_judgment_basis": ["理由1", "理由2", "理由3", "理由4"],
                },
            },
        },
    }

    sections = [
        "请对以下通话对话进行四步推理分析，严格按下方 JSON 骨架返回结果，不输出任何额外文字。",
        "【对话文本】\n" + dialogue_text,
        (
            "【输出结构约束】\n"
            "· Step2.output 必须含 phase_system + phases（每项含 id/type/turn_range/description）\n"
            "· Step3.output 必须含 signal_type + signals（每项含 signal_id/phase/description）\n"
            "· phases.id 应使用原始 phase_id（同一 type 可出现多次，必须可区分实例）\n"
            "· signal_type 与 phase_system 严格对应：fraud_6phase→fraud_tactics，normal_3phase→service_behaviors\n"
            "· is_fraud=true  → Step4 含 fraud_type/confidence/disambiguation/reasoning_summary\n"
            "· is_fraud=false → Step4 含 call_category/confidence/normal_judgment_basis（列表≥4条）\n"
            "· 每个 step.think 必须为 5-8 句中文推理，且至少包含2处来自对话的证据（turn 区间或关键原话）\n"
            "· 两套 Step4 字段互斥，不得混用；仅输出 JSON，不得有注释或额外顶层字段"
        ),
        _json_block("【请严格遵循以下 JSON 字段骨架（字段名保持一致）】", output_schema),
        (
            "【Step4 正常样本替换规则】\n"
            "当 is_fraud=false 时，step_4_judgment.output 不得包含 fraud_type/disambiguation/reasoning_summary，"
            "必须包含 call_category 与 normal_judgment_basis。"
        ),
        (
            "【阶段对齐约束】\n"
            "若标签约束中提供 signals_with_phase（每项含 signal_id + phase），"
            "则 Step3 中对应 signal_id 必须放在同一 phase，不得错位。"
        ),
    ]
    if label_hints:
        sections.append(
            _json_block(
                "【已有人审标签约束（必须保持一致，不允许与下列字段冲突）】",
                label_hints,
            )
        )
    return "\n\n".join(sections)


def build_repair_prompt(
    dialogue_text: str,
    invalid_json_text: str,
    validation_errors: list[str],
    label_hints: Optional[Dict[str, Any]],
) -> str:
    payload: Dict[str, Any] = {
        "dialogue_text": dialogue_text,
        "invalid_output": invalid_json_text,
        "validation_errors": validation_errors,
        "instruction": (
            "请逐条对照 validation_errors 修复以下输出，使其符合四步 evidence_chain 规范。"
            "要求：① 仅输出修复后的完整 JSON 对象，不要输出解释或 Markdown 代码块；"
            "② 修复时不得改变已有推理结论（think/output 语义），仅补全缺失字段或修正格式；"
            "③ signal_type 必须与 phase_system 保持对应；"
            "④ is_fraud=true 与 is_fraud=false 的 Step4 字段集合互斥，不可混用；"
            "⑤ think 字段必须补足为高质量推理（5-8句，包含对话证据）。"
        ),
    }
    if label_hints:
        payload["label_hints"] = label_hints
    return _json_block("请修复以下输出：", payload)
