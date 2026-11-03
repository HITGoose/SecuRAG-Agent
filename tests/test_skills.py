"""
阶段3 单元测试：技能目录解析、白名单、长度上限。
运行：python -m unittest tests.test_skills -v
"""
import tempfile
import unittest
from pathlib import Path

from shop_tools import prepare_args, register_schema
from skill_loader import MAX_SKILL_CHARS, SkillLibrary, load_skill_schema


def write_skill(root: Path, dirname: str, text: str) -> None:
    (root / dirname).mkdir(parents=True)
    (root / dirname / "SKILL.md").write_text(text, encoding="utf-8")


class TestSkillLibrary(unittest.TestCase):
    def test_repo_skills_parse(self):
        lib = SkillLibrary()
        self.assertEqual(sorted(lib.names), ["complaint", "return_exchange"])
        self.assertIn("return_exchange：", lib.catalog())
        self.assertNotIn("## 1.", lib.catalog())  # 目录里只有描述，没有正文
        self.assertIn("# 退换货流程", lib.load("return_exchange")["content"])

    def test_load_skill_args_limited_to_whitelist(self):
        lib = SkillLibrary()
        register_schema(load_skill_schema(lib.names))
        self.assertEqual(prepare_args("load_skill", {"name": "complaint"}), ({"name": "complaint"}, None))
        for bad in ("../../.env", "skills/complaint/SKILL.md", "refund_now", ""):
            self.assertIsNotNone(prepare_args("load_skill", {"name": bad})[1], bad)
        with self.assertRaises(KeyError):
            lib.load("../complaint")

    def test_bad_skill_files_rejected(self):
        cases = {
            "no_header": "# 没有头信息",
            "missing_desc": "---\nname: missing_desc\nversion: v1\n---\n正文",
            "name_mismatch": "---\nname: other\ndescription: d\nversion: v1\n---\n正文",
            "too_long": "---\nname: too_long\ndescription: d\nversion: v1\n---\n" + "字" * (MAX_SKILL_CHARS + 1),
        }
        for dirname, text in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                write_skill(Path(tmp), dirname, text)
                with self.assertRaises(ValueError, msg=dirname):
                    SkillLibrary(Path(tmp))


if __name__ == "__main__":
    unittest.main()
