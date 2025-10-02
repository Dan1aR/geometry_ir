"""Reference helpers for the GeoScript intermediate representation."""

from textwrap import dedent

BNF = dedent(
    """
    Program   := { Stmt }
    Stmt      := Scene | Layout | Points | Obj | Placement | Annot | Target | Rules | Comment

    Scene     := 'scene' STRING
    Layout    := 'layout' 'canonical=' ID 'scale=' NUMBER
    Points    := 'points' ID { ',' ID }

    Annot     := 'label point' ID Opts?
              | 'sidelabel' Pair STRING Opts?

    Target    := 'target'
                 ( 'angle' 'at' ID 'rays' Pair Pair
                 | 'length' Pair
                 | 'point' ID
                 | 'circle' '(' STRING ')'
                 | 'area' '(' STRING ')'
                 | 'arc' ID '-' ID 'on' 'circle' 'center' ID Opts?
                 )

    Obj       := 'segment' Pair Opts?
               | 'ray'     Pair Opts?
               | 'line'    Pair Opts?
               | 'circle' 'center' ID ('radius-through' ID | 'tangent' '(' EdgeList ')' ) Opts?
               | 'circle' 'through' '(' IdList ')' Opts?
               | 'circumcircle' 'of' IdChain Opts?
               | 'incircle'    'of' IdChain Opts?
               | 'perpendicular' 'at' ID 'to' Pair 'foot' ID Opts?
               | 'parallel' 'through' ID 'to' Pair Opts?
               | 'median'  'from' ID 'to' Pair 'midpoint' ID Opts?
               | 'angle' 'at' ID 'rays' Pair Pair Opts?
               | 'right-angle' 'at' ID 'rays' Pair Pair Opts?
               | 'equal-segments' '(' EdgeList ';' EdgeList ')' Opts?
               | 'parallel-edges' '(' Pair ';' Pair ')' Opts?
               | 'tangent' 'at' ID 'to' 'circle' 'center' ID Opts?
               | 'diameter' Pair 'to' 'circle' 'center' ID Opts?
               | 'line' ID '-' ID 'tangent' 'to' 'circle' 'center' ID 'at' ID Opts?
               | 'polygon' IdChain Opts?
               | 'triangle' ID '-' ID '-' ID Opts?
               | 'quadrilateral' ID '-' ID '-' ID '-' ID Opts?
               | 'parallelogram' ID '-' ID '-' ID '-' ID Opts?
               | 'trapezoid' ID '-' ID '-' ID '-' ID Opts?
               | 'rectangle' ID '-' ID '-' ID '-' ID Opts?
               | 'square' ID '-' ID '-' ID '-' ID Opts?
               | 'rhombus' ID '-' ID '-' ID '-' ID Opts?

    Placement := 'point' ID 'on' Path
               | 'intersect' '(' Path ')' 'with' '(' Path ')' 'at' ID (',' ID)? Opts?
               | 'midpoint' ID 'of' Pair Opts?
               | 'foot' ID 'from' ID 'to' Pair Opts?

    Path      := 'line'    Pair
                | 'ray'     Pair
                | 'segment' Pair
                | 'circle' 'center' ID
                | 'angle-bisector' 'at' ID 'rays' Pair Pair ('external')?
                | 'median'  'from' ID 'to' Pair
                | 'perpendicular' 'at' ID 'to' Pair

    EdgeList  := Pair { ',' Pair }
    IdList    := ID { ',' ID }
    IdChain   := ID '-' ID { '-' ID }
    Pair      := ID '-' ID

    Opts      := '[' KeyVal { ' ' KeyVal } ']'
    KeyVal    := KEY '=' (VALUE | STRING)
    """
).strip()

_PROMPT_CORE = dedent(
    """
    ROLE & INPUT
    - You are a *Geometry Scene Writer*. Convert a short RU/EN description of a Euclidean construction into
      a faithful **GeoScript** scene. Model only what the text states (objects, relations, targets). Do **NOT** solve
      or invent coordinates, measures, or helper elements beyond what the prompt allows.

    DELIVERY FORMAT
    - Respond with GeoScript wrapped exactly once in &lt;geoscript> ... &lt;/geoscript>. No prose, no Markdown fences, no JSON.
    - Emit one statement per line. Comments use '#'. Stick to ASCII; write '^\\circ' instead of the ° symbol.
    - Prefer textual annotations over equations. E.g. "AB = 4" becomes `segment A-B [length=4]` or
      `sidelabel A-B "4"` (do NOT embed '=' inside the quotes).

    AUTHORING WORKFLOW
    1. Declare the scene header before any construction:
         scene "Problem title"
         layout canonical=<id> scale=<number>
         points A, B, C[, ...]      # list every named point, comma-separated
    2. Translate the givens into explicit GeoScript statements, mirroring the clean samples in tests/integrational/gir/*.gir.
       - Use one command per fact: segments, polygons, circles, parallels, right angles, tangents, equalities, etc.
       - Circles have two syntaxes:
         * Known center ➜ `circle center O radius-through B` (optionally add tangency Opts). Put extra on-circle points with
           separate `point X on circle center O` lines.
         * Unknown center ➜ `circle through (A, B, C)` (or other point counts). Do **not** mix `center` with `through (...)`.
       - Tangents must be explicit. RU cues like "касательная" and EN "tangent" map to:
         * Tangent segment from an external point ➜ declare the segment (`segment A-B`) **and** the tangency (`line A-B tangent to circle center O at B`).
         * Touchpoint-specified tangent line without the external point ➜ `tangent at B to circle center O`.
         * A circle tangent to sides ➜ `circle center O tangent (A-B, C-D)`.
         Anchor the touchpoints on the circle separately when the prompt implies it.
       - Keep constraints declarative: place points with `point ... on ...` or `intersect (...) with (...)`.
    3. Add annotations (labels, side texts) the prompt requires and finish with `target ...` lines capturing what to find.
    4. Include a `rules [...]` line only when the problem explicitly restricts solving/auxiliary work; omit it otherwise.

    OPTIONS CHEATSHEET
    - GeoScript only reads the options listed below. Never output bare `[]`, and never invent new keys.
      * Global rules: `no_solving`, `allow_auxiliary`, `no_unicode_degree`, `no_equations_on_sides`, `mark_right_angles_as_square`.
      * Segment/edge data: `length=<number|sqrt(number)|number*sqrt(number)>`, `label="text"`.
        Square roots must use `sqrt(...)` with parentheses (e.g., `[length=sqrt(5)]` or `[length=3*sqrt(2)]`). Do not use the
        legacy LaTeX-style backslash-sqrt-with-braces form.
      * Polygon metadata: `bases=A-D` for trapezoids, `isosceles=atB` for isosceles triangles.
      * Angle/arc data: `degrees=<number>`, `label="text"`, `mark=square` for right angles.
      * Point/line markers: `mark=midpoint`, `mark=directed`, `color=<name>` when the prompt specifies styling.
      * Text positioning: `pos=left|right` for `sidelabel`, optional `label="text"` on `target` and equality statements.
    - If a prompt does not demand an option, omit the brackets entirely.

    SANITY CHECKS BEFORE SENDING
    - Every identifier used in the body appears in the `points` list (case-insensitive match).
    - Each statement matches the BNF (see below) and any options stay inside `[key=value ...]` with valid keys.
    - Circles, parallels, right angles, tangencies ("касательная"), and perpendiculars are explicitly declared when the text implies them.
    - Use `right-angle ... [mark=square]` or a perpendicular construction whenever the prompt enforces a 90° relation.
    - If the text says a figure is cyclic/inscribed/circumscribed, add the corresponding circle statement.

    STYLE GUARDRAILS
    - Minimal, declarative GeoScript; no calculations, no helper geometry unless explicitly allowed.
    - Avoid unicode √ or raw equations inside labels; use TeX-style text like `"\\sqrt{3}"` when necessary.
    - Follow the BNF for statement order and syntax. Options go inside square brackets separated by spaces.

    FEW-SHOT GUIDANCE (INPUT ➜ OUTPUT)
    - Example 1
      Input: "Triangle ABC has angles 38°, 110°, 32°. Points D and E lie on AC with D on AE, BD = DA, BE = EC. Find angle DBE."
      Output:
      <geoscript>
      scene "Triangle with given angles; BD=DA; BE=EC; find angle DBE"
      layout canonical=triangle_ABC scale=1
      points A, B, C, D, E
      triangle A-B-C
      angle at A rays A-B A-C [degrees=38]
      angle at B rays B-A B-C [degrees=110]
      angle at C rays C-A C-B [degrees=32]
      segment A-C
      point D on segment A-C
      point E on segment A-C
      point D on segment A-E
      segment B-D
      segment D-A
      segment B-E
      segment E-C
      equal-segments (B-D ; D-A) [label="given"]
      equal-segments (B-E ; E-C) [label="given"]
      target angle at B rays B-D B-E [label="?DBE"]
      </geoscript>

    - Example 2
      Input: "In trapezoid ABCD, AD is the base, CD = 12 cm. Diagonals intersect at O. The distance from O to CD is 5 cm. Find the area of triangle AOB."
      Output (lines to place inside the geoscript wrapper shown above):
      scene "Trapezoid diagonals area"
      layout canonical=generic scale=1
      points A, B, C, D, O, M
      trapezoid A-B-C-D [bases=A-D]
      segment C-D [length=12]
      intersect (segment A-C) with (segment B-D) at O
      perpendicular at O to C-D
      intersect (perpendicular at O to C-D) with (segment C-D) at M
      segment O-M [length=5]
      target area ("Find area of triangle AOB")

    - Example 3
      Input: "Right triangle ABC has angle B = 21 degrees. Let CD be the bisector and CM the median from the right vertex C. Find the angle between CD and CM."
      Output (lines to place inside the geoscript wrapper shown above):
      scene "Right-angled triangle with angle B=21^\\circ, find angle between CD and CM"
      layout canonical=triangle_ABC scale=1.0
      points A, B, C, D, M
      triangle A-B-C
      angle at C rays C-A C-B [degrees=90]
      angle at B rays B-A B-C [degrees=21]
      angle at A rays A-B A-C [degrees=69]
      intersect (angle-bisector at C rays C-A C-B) with (segment A-B) at D
      intersect (median from C to A-B) with (segment A-B) at M
      target angle at C rays C-D C-M [label="?"]

    - Example 4
      Input: "Circle with center O has diameter AB. Points C and D lie on the circle. Find angle ACB."
      Output (lines to place inside the geoscript wrapper shown above):
      scene "Circle with diameter AB; find angle ACB"
      layout canonical=generic scale=1
      points A, B, C, D, O
      segment A-B
      circle center O radius-through A
      diameter A-B to circle center O
      point C on circle center O
      point D on circle center O
      target angle at C rays C-A C-B [label="?ACB"]
    """
).strip()


def get_llm_prompt(*, include_bnf: bool = True) -> str:
    """Return the standard GeometryIR prompt for LLM agents."""
    sections = [_PROMPT_CORE]
    if include_bnf:
        sections.append("SYNTAX REFERENCE (BNF)\n" + BNF)
    return "\n\n".join(sections)


LLM_PROMPT = get_llm_prompt()

__all__ = ["BNF", "LLM_PROMPT", "get_llm_prompt"]
