"""
业务技能（Skills）：skills/<name>/SKILL.md，开头是 YAML 头信息（name / description / version），正文是完整流程。

两种用法（ShopAgent 的 skills 参数）：
- "full"：所有技能的正文都放进系统提示（全文常驻）
- "ondemand"：系统提示只放每个技能的一句话描述；模型需要时调用 load_skill(name) 拿正文（渐进加载）

安全：只认启动时从仓库 skills/ 目录读到的技能名（白名单），load_skill 只接受这些名字，
不接受路径，也不会从商品说明、用户消息等文本里加载技能；技能不带来任何新工具或新权限。
"""
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"
MAX_SKILL_CHARS = 3000  # 单个技能正文上限，防止某个技能把上下文撑爆


def _parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} 缺少 YAML 头信息")
    head, body = text[4:].split("\n---\n", 1)
    meta = dict(line.split(":", 1) for line in head.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    for key in ("name", "description", "version"):
        if not meta.get(key):
            raise ValueError(f"{path} 头信息缺少 {key}")
    if meta["name"] != path.parent.name:
        raise ValueError(f"{path} 的 name 与目录名不一致")
    body = body.strip()
    if len(body) > MAX_SKILL_CHARS:
        raise ValueError(f"{path} 正文超过 {MAX_SKILL_CHARS} 字")
    return {**meta, "content": body}


class SkillLibrary:
    def __init__(self, root: Path = SKILLS_DIR):
        self.skills = {}
        for path in sorted(root.glob("*/SKILL.md")):
            skill = _parse(path)
            self.skills[skill["name"]] = skill

    @property
    def names(self) -> list:
        return list(self.skills)

    def catalog(self) -> str:
        """渐进加载时放进系统提示的部分：每个技能一行描述。"""
        return "\n".join(f"- {s['name']}：{s['description']}" for s in self.skills.values())

    def full_text(self, names=None) -> str:
        """全文常驻时放进系统提示的部分；也用于把本会话已加载过的技能带到后续轮次。"""
        picked = [self.skills[n] for n in (names if names is not None else self.names)]
        return "\n\n".join(s["content"] for s in picked)

    def load(self, name: str) -> dict:
        """load_skill 工具的实现。name 已经被 schema 的枚举限制在白名单内，这里再检查一次。"""
        if name not in self.skills:
            raise KeyError(name)
        s = self.skills[name]
        return {"name": s["name"], "version": s["version"], "content": s["content"]}


def load_skill_schema(names: list) -> dict:
    return {"type": "function", "function": {
        "name": "load_skill",
        "description": "加载一个业务技能的完整流程说明。处理退换货、投诉等业务前，先加载对应技能，再按流程调用其他工具。",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["name"],
                       "properties": {"name": {"type": "string", "enum": names, "description": "技能名"}}}}}
