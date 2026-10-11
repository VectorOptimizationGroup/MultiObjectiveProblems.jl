# Problem Families

MultiObjectiveProblems.jl groups benchmark constructors by their source family or
publication. Each constructor returns a
[`MultiObjectiveProblem`](@ref MultiObjectiveProblems.MultiObjectiveProblem)
instance with effective dimensions, objective and constraint evaluators,
variable and constraint bounds when available, and registered analytical
derivatives when implemented.

## Documented families

- [Amaral–Assunção–Souza (AAS)](@ref): two fixed-dimension problems
  from Amaral, Assunção, and Souza
  [AAS2025](@cite).
- [Ansary–Panda (AP)](@ref): four fixed-dimension problems from
  Md. A. T. Ansary and G. Panda
  [AP2014](@cite).
- [Binh–Korn (BK)](@ref): one fixed-dimension problem from To Thanh Binh and
  Ulrich Korn
  [BK1996](@cite).
- [Das–Dennis (DD)](@ref): one constrained, fixed-dimension problem from
  Indraneel Das and J. E. Dennis.
  [DD1998](@cite).
- [Dumitrescu-Grosan-Oltean (DGO)](@ref): three fixed-dimension problems from
  Crina Grosan, Dan Dumitrescu, and Mihai Oltean
  [DGO2000](@cite).
- [Deb–Thiele–Laumanns–Zitzler (DTLZ)](@ref): scalable test problems from Deb,
  Thiele, Laumanns, and Zitzler
  [DTLZ2005](@cite).
- [Farhang-Mehr-Azarm (FA)](@ref): one fixed-dimension problem from
  A. Farhang-Mehr and S. Azarm
  [FA2002](@cite).
- [Farina (Far)](@ref): one fixed-dimension problem originating in M. Farina
  [Far2002](@cite), using the corrected formulation cataloged by Huband et
  al. [Huband2006](@cite).
- [Fliege–Drummond–Svaiter (FDS)](@ref): one scalable, three-objective problem
  introduced by J. Fliege, L. M. Graña Drummond, and B. F. Svaiter
  [FDS2009](@cite).
- [Fonseca–Fleming (FF)](@ref): one fixed-dimension problem from Carlos M.
  Fonseca and Peter J. Fleming [FF1995](@cite).
- [Hillermeier (Hil)](@ref): one unconstrained, fixed-dimension academic
  problem from C. Hillermeier [Hil2001](@cite).
- [Ikeda–Kita–Kobayashi (IKK)](@ref): one fixed-dimension problem from
  K. Ikeda, H. Kita, and S. Kobayashi [IKK2001](@cite).
- [Ishibuchi–Murata (IM)](@ref): one fixed-dimension problem from H. Ishibuchi
  and T. Murata [IM1998](@cite).
- [Jin–Olhofer–Sendhoff (JOS)](@ref): two variable-dimension problems from
  Yaochu Jin, Markus Olhofer, and Bernhard Sendhoff [JOS2001](@cite), using
  source-order numbering and documenting the alternative Huband catalog name.
- [Kim–de Weck (KW)](@ref): one fixed-dimension problem from the second
  numerical example of I. Y. Kim and O. L. de Weck [KW2005](@cite), expressed
  in the package's minimization convention.
- [Lis–Eiben (LE)](@ref): one fixed-dimension problem from Test 1 of J. Lis and
  A. E. Eiben [LE1997](@cite).
- [Lovison (Lov)](@ref): six fixed-dimension examples from Alberto Lovison
  [Lovison2011](@cite), including a smooth regularization of ZDT3
  [ZDT2000](@cite), with documented minimization conventions.
- [Laumanns–Thiele–Deb–Zitzler (LTDZ)](@ref): one fixed-dimension,
  three-objective problem from Laumanns, Thiele, Deb, and Zitzler
  [Laumanns2002](@cite), using the formulation and `LTDZ1` name given in
  Table XVI of Huband et al. [Huband2006](@cite).
- [Moré–Garbow–Hillstrom (MGH)](@ref): four problems derived from the
  nonlinear least-squares collection of Moré, Garbow, and Hillstrom
  [MGH1981](@cite), using duly documented extensions of the multiobjective
  formulations based on Mita, Fukuda, and Yamashita [Mita2019](@cite).
- [Mao–Hirasawa–Hu–Murata (MHHM)](@ref): two fixed-dimension,
  three-objective problems from the numerical simulations of Jiangming Mao,
  K. Hirasawa, Jinlu Hu, and J. Murata [MHHM2000](@cite).
- [Molyneaux–Leyland–Favrat (MLF)](@ref): two fixed-dimension problems from
  A. K. Molyneaux, G. B. Leyland, and D. Favrat [MLF2001](@cite), including
  the corrected `MLF1` formulation cataloged by Huband et al.
  [Huband2006](@cite).
- [Miglierina–Molho–Recchioni (MMR)](@ref): four fixed-dimension problems from
  Tests 1–4 of E. Miglierina, E. Molho, and M. C. Recchioni
  [MMR2008](@cite).
- [Preuss–Naujoks–Rudolph (PNR)](@ref): the fixed-dimension Case 1
  specialization of the `TWO-ON-ONE` test problem by Mike Preuss, Boris
  Naujoks, and Günter Rudolph [PNR2006](@cite).
- [Quagliarella–Vicini (QV)](@ref): one variable-dimension problem introduced
  by D. Quagliarella and A. Vicini [QV1998](@cite) and subsequently cataloged
  by Huband et al. [Huband2006](@cite).
- [Stadler–Dauer (SD)](@ref): one fixed-dimension truss-design problem from the
  “Elastic Trusses” example of W. Stadler and J. Dauer [SD1992](@cite),
  implemented in its normalized instance.
- [Socha–Kisiel-Dorohinicki (SK)](@ref): two fixed-dimension problems used as
  test cases by K. Socha and M. Kisiel-Dorohinicki [SK2002](@cite), including
  the corrected `SK1` formulation cataloged by Huband et al.
  [Huband2006](@cite), with documented minimization conventions.
- [Schütze–Laumanns–Coello Coello–Dellnitz–Talbi (SLCDT)](@ref): two
  fixed-dimension problems from Oliver Schütze, Marco Laumanns, Carlos A.
  Coello Coello, Michael Dellnitz, and El-Ghazali Talbi [SLCDT2008](@cite),
  with the perturbation coefficient `λ` of `SLCDT1` exposed as a constructor
  parameter.
- [Sefrioui–Perlaux (SP)](@ref): one fixed-dimension problem from the simple
  mathematical example of M. Sefrioui and J. Perlaux [SP2000](@cite).
- [Shim–Suh–Furukawa–Yagawa–Yoshimura (SSFYY)](@ref): the fixed-dimension
  Test problem 2 of Mun-Bo Shim, Myung-Won Suh, Tomonari Furukawa, Genki
  Yagawa, and Shinobu Yoshimura [SSFYY2002](@cite).
- [Tan–Khor–Lee–Yang (TKLY)](@ref): the fixed-dimension Test problem 2 of
  K. C. Tan, E. F. Khor, T. H. Lee, and Y. J. Yang [TKLY2003](@cite).
- [Toint (Toi)](@ref): four problems derived from the partially separable
  collection of Ph. L. Toint [Toi1983](@cite), whose element functions are
  taken as objectives in the multiobjective adaptations of Mita, Fukuda, and
  Yamashita [Mita2019](@cite).
- [Valenzuela-Rendón–Uresti-Charre (VU)](@ref): two fixed-dimension problems
  from M. Valenzuela-Rendón and E. Uresti-Charre [VU1997](@cite), using the
  formulations cataloged by Huband et al. [Huband2006](@cite).
- [Van Veldhuizen (VV)](@ref): five problems of the test suite
  assembled by David A. Van Veldhuizen [VV1999](@cite) and cataloged by Huband
  et al. [Huband2006](@cite).
- [Zitzler–Deb–Thiele (ZDT)](@ref): five variable-dimension, two-objective
  problems from the real-valued test functions of Eckart Zitzler, Kalyanmoy
  Deb, and Lothar Thiele [ZDT2000](@cite); the source's binary-encoded test
  function is not implemented.
- [Zitzler–Laumanns–Thiele (ZLT)](@ref): one independent-dimension problem
  from the test suite of Eckart Zitzler, Marco Laumanns, and Lothar Thiele
  [ZLT2001](@cite), where it is named `SPH-m`, using the `ZLT1` name and the
  general objective count cataloged by Huband et al. [Huband2006](@cite).
