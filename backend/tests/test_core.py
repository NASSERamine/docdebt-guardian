from pathlib import Path

from docdebt.core.auditor import Auditor, AuditorConfig, audit_javadoc
from docdebt.core.diff import parse_unified_diff
from docdebt.core.extractor import Extractor, ExtractorConfig
from docdebt.core.java_parser import (
    extract_signature_from_method_code,
    invoked_method_names,
    parse_java_source,
    parse_method_snippet,
)
from docdebt.core.javadoc import parse_javadoc
from docdebt.core.models import Severity, Visibility

from .conftest import KELVIN_SOURCE


def test_parse_methods_signature_and_javadoc():
    units = parse_java_source(KELVIN_SOURCE, "Temperature.java")
    names = [u.name for u in units]
    assert names == ["toCelsius", "pick", "read", "helper"]
    to_celsius = units[0]
    assert to_celsius.signature.param_names == ["tempInKelvin"]
    assert to_celsius.signature.return_type == "double"
    assert to_celsius.javadoc.startswith("/**")
    assert to_celsius.javadoc_start_line == 5
    assert to_celsius.start_line == 10
    assert to_celsius.visibility == Visibility.PUBLIC
    assert units[2].signature.throws == ("java.io.IOException",)
    assert units[3].javadoc is None and units[3].visibility == Visibility.PRIVATE


def test_bodiless_method_does_not_swallow_next_method():
    src = "abstract class A {\n  /** a */ abstract int f(int x);\n  /** b */ void g() {}\n}"
    f, g = parse_java_source(src)
    assert f.code == "abstract int f(int x);"
    assert not f.has_body and g.has_body


def test_varargs_arrays_generics_nested_and_interface():
    src = (
        "interface I { /** d */ <T> java.util.List<T> m(String... args, int v[], Map<K, V> m); }\n"
        "class O { class In { /** d */ O(int z) {} } }"
    )
    m, ctor = parse_java_source(src)
    assert [p.type for p in m.signature.parameters] == ["String...", "int[]", "Map<K, V>"]
    assert m.visibility == Visibility.PUBLIC  # interface member
    assert ctor.class_name == "O.In" and ctor.signature.is_constructor


def test_snippet_and_research_compatible_signature():
    unit = parse_method_snippet("/** x */\npublic int add(int x) { return x; }")
    assert unit.javadoc == "/** x */" and unit.name == "add"
    assert extract_signature_from_method_code("int broken( { return 1; }") is None
    sig = extract_signature_from_method_code("public Foo(int a) throws X {}")
    assert sig.is_constructor and sig.is_void and sig.param_names == ["a"]


def test_invoked_names():
    assert invoked_method_names("void a(){ b(); this.c(1); X.d(); }") == {"b", "c", "d"}


def test_parse_javadoc_matches_research_behaviour():
    parsed = parse_javadoc("/**\n * Desc.\n * @param a first\n * @return r\n * @throws java.io.IOException x\n */")
    assert parsed.description == "Desc."
    assert parsed.params == {"a": "first"}
    assert parsed.return_desc == "r"
    assert parsed.throws == {"IOException": "x"}
    assert parse_javadoc(None).description == ""


def test_auditor_rules():
    units = {u.name: u for u in parse_java_source(KELVIN_SOURCE, "T.java")}
    auditor = Auditor()
    rules = {f.rule for f in auditor.audit(units["toCelsius"]).findings}
    assert rules == {"missing-param", "unknown-param"}
    assert auditor.audit(units["toCelsius"]).has_structural_debt
    unknown = next(f for f in auditor.audit(units["toCelsius"]).findings if f.rule == "unknown-param")
    assert unknown.line == 7  # the line of "@param tempInCelsius"
    assert not auditor.audit(units["pick"]).findings  # semantic debt only: invisible here
    read = auditor.audit(units["read"])
    assert [f.rule for f in read.errors] == ["missing-throws"]
    relaxed = Auditor(AuditorConfig(require_throws=False)).audit(units["read"])
    assert not relaxed.has_structural_debt


def test_auditor_warnings_never_count_as_debt():
    sig = parse_method_snippet("void f(int a) {}").signature
    findings = audit_javadoc("/** d\n @param a\n @param a\n @return x */", sig, "id")
    assert {f.rule for f in findings} == {"duplicate-param", "empty-param", "return-on-void"}
    assert all(f.severity == Severity.WARNING for f in findings)


def test_diff_changed_lines():
    diff = """diff --git a/src/A.java b/src/A.java
--- a/src/A.java
+++ b/src/A.java
@@ -10,3 +10,4 @@ class A {
 context
-old line
+new line
+added
 context
diff --git a/B.java b/B.java
--- a/B.java
+++ /dev/null
@@ -1,2 +0,0 @@
-x
-y
"""
    changes = parse_unified_diff(diff)
    assert list(changes) == ["src/A.java"]
    assert changes["src/A.java"].changed_lines == {11, 12}


def test_extractor_from_diff(tmp_path: Path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "Temperature.java").write_text(KELVIN_SOURCE, encoding="utf-8")
    diff = """--- a/demo/Temperature.java
+++ b/demo/Temperature.java
@@ -20,1 +20,1 @@
-    public int pick(int a, int b) {
+    public int pick(int a, int b) {
"""
    units = Extractor().from_diff(diff, tmp_path)
    assert [u.name for u in units] == ["pick"]
    assert units[0].file_path == "demo/Temperature.java"


def test_extractor_filters(tmp_path: Path):
    (tmp_path / "gen").mkdir()
    (tmp_path / "T.java").write_text(KELVIN_SOURCE, encoding="utf-8")
    (tmp_path / "gen" / "G.java").write_text(KELVIN_SOURCE, encoding="utf-8")
    cfg = ExtractorConfig(exclude=("gen/*",), visibilities=frozenset({Visibility.PUBLIC}))
    units = Extractor(cfg).from_directory(tmp_path)
    assert {u.file_path for u in units} == {"T.java"}
    assert [u.name for u in units] == ["toCelsius", "pick", "read"]  # helper: private, no doc


def test_explicit_paths_are_never_skipped_silently(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError, match="Path not found"):
        Extractor().from_paths([tmp_path / "missing.java"])
    notes = tmp_path / "notes.md"
    notes.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Not a .java file"):
        Extractor().from_paths([notes])
    assert Extractor().from_paths([tmp_path]) == []  # an existing but empty directory is fine
