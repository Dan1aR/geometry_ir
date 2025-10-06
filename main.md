# geoscript-ir

**GeoScript IR** is a parser, validator, and desugarer for a small DSL describing **2D Euclidean geometry scenes**.
It turns human-readable problem statements into a canonical intermediate representation that can later feed a numeric solver for automatic **nice** points positioning and TikZ generation.

## BNF

```
Program   := { Stmt }
Stmt      := Scene | Layout | Points | Obj | Placement | Annot | Target | Rules | Comment

Scene     := 'scene' STRING
Layout    := 'layout' 'canonical=' ID 'scale=' NUMBER
Points    := 'points' ID { ',' ID }

Annot     := 'label point' ID Opts?
          | 'sidelabel' Pair STRING Opts?

Target    := 'target'
             ( 'angle' Angle3
             | 'length' Pair
             | 'point' ID
             | 'circle' '(' STRING ')'
             | 'area' '(' STRING ')'
             | 'arc' ID '-' ID 'on' 'circle' 'center' ID Opts?
             )

Obj       := 'segment' Pair Opts?
           | 'ray'     Pair Opts?
           | 'line'    Pair Opts?
           | 'circle' 'center' ID 'radius-through' ID Opts?
           | 'circle' 'through' '(' IdList ')' Opts?
           | 'circumcircle' 'of' IdChain Opts?
           | 'incircle'    'of' IdChain Opts?
           | 'perpendicular' 'at' ID 'to' Pair 'foot' ID Opts?
           | 'parallel' 'through' ID 'to' Pair Opts?
           | 'median'  'from' ID 'to' Pair 'midpoint' ID Opts?
           | 'angle' Angle3 Opts?
           | 'right-angle' Angle3 Opts?
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
            | 'angle-bisector' Angle3 ('external')?
            | 'median'  'from' ID 'to' Pair
            | 'perpendicular' 'at' ID 'to' Pair

EdgeList  := Pair { ',' Pair }
IdList    := ID { ',' ID }
IdChain   := ID '-' ID { '-' ID }
Pair      := ID '-' ID
Angle3    := ID '-' ID '-' ID

Opts      := '[' KeyVal { ' ' KeyVal } ']'
KeyVal    := KEY '=' (VALUE | STRING)

```

### Numeric solver (GeometryIR → SciPy)

The `geoscript_ir.solver` module compiles validated GeoScript into a
numeric model and optimizes the residuals with `scipy.optimize.least_squares`.
