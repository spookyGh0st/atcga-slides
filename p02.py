import numpy as np
from manim import *
from manim_slides.slide import Slide
from dataclasses import dataclass
from sympy import symbols
import sympy as sympy

N = 4
basis_colors = [PURPLE, ORANGE, GREEN, MAROON]

@dataclass
class polynomial:
    # Name
    n: str
    # coefficients
    c: list[float]
    col: color

def polynom_tex(p: polynomial):
    t = [p.n, " := "]
    for i in range(len(p.c)):
        if (i > 0): t.append('+')
        t.append(str(p.c[i]))
        t.append('\cdot')
        if (i == 0):
            t.append('1')
        elif (i == 1):
            t.append('t')
        else:
            t.append('t^' + str(i))

    tex = MathTex(*t)
    tex[0].set_color(p.col)
    for i in range(len(p.c)):
        tex[i*4+4][:].set_color(basis_colors[i])
    return tex

l_axes = Axes(
    x_range=[-0.5, 4.5, 1],
    y_range=[-0.5, 2.5, 1],
    x_length=5,
    y_length=2.5,
    x_axis_config={
        "numbers_to_include": np.arange(-0, 4.01, 2),
        "numbers_with_elongated_ticks": np.arange(-0, 4.01, 2),
    },
    y_axis_config={
        "numbers_to_include": np.arange(-0, 2.01, 2),
        "numbers_with_elongated_ticks": np.arange(-0, 2.01, 2),
    },
    tips=False,
).to_corner(LEFT + UP)

number_plane = NumberPlane(
    x_range=[0.0, 5.2, 1],
    y_range=[-2.6, 2.6, 1],
    x_length=5,
    y_length=5.0,
    background_line_style={
            "stroke_color": TEAL,
            "stroke_width": 4,
            "stroke_opacity": 0.6
    }
).to_corner(LEFT+UP)

r_axes = Axes(
    x_range=[-5, 5.01, 1],
    y_range=[-5, 5.01, 1],
    x_length=6,
    y_length=6,
    x_axis_config={
        "numbers_to_include": np.arange(-5, 5.01, 5),
        "numbers_with_elongated_ticks": np.arange(-5, 5.01, 5),
    },
    y_axis_config={
        "numbers_to_include": np.arange(-5, 5.01, 2),
        "numbers_with_elongated_ticks": np.arange(-5, 5.01, 5),
    },
    tips=False,
).to_edge(RIGHT)

r_axes_norm = Axes(
    x_range=[-0.5, 1.51, 1],
    y_range=[-0.5, 1.51, 1],
    x_length=6,
    y_length=6,
    x_axis_config={
        "numbers_to_include": np.arange(-0.0, 1.51, 1),
        "numbers_with_elongated_ticks": np.arange(-0.0, 1.51, 1),
    },
    y_axis_config={
        "numbers_to_include": np.arange(-0.0, 1.51, 1),
        "numbers_with_elongated_ticks": np.arange(-0.0, 1.51, 1),
    },
    tips=False,
).to_edge(RIGHT)

def plot_pol_r(p: polynomial, axes = r_axes):
    def f(t):
        x = 0
        for i in range(len(p.c)):
            x += p.c[i] * (t**i)
        return x
    return axes.plot(lambda t: f(t),color=p.col)
def plot_pol_basis_r(p: polynomial, axes = r_axes):
    def f(t):
        x = 0
        for i in range(len(p.c)):
            x += p.c[i] * (t**i)
        return x
    return axes.plot(lambda t: f(t),color=p.col,stroke_width=DEFAULT_STROKE_WIDTH/2)

buff1 = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
buff2 = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2

#=======================================
#   Lagrange
#=======================================
def lagrange_coefficients(n, i):
    x = symbols('x')
    numerator = 1
    denominator = 1
    for m in range(n):
        if m != i:
            numerator *= (x - m)
            denominator *= (i - m)

    basis_polynomial = numerator / denominator
    c = basis_polynomial.as_poly().all_coeffs()
    c.reverse()
    return c



def lagrange_basis(i, n):
    c = lagrange_coefficients(n, i)
    return polynomial("e_" + str(i), c, basis_colors[i])


lagrange_polynomials = [
    lagrange_basis(i, N)
    for i in range(N)
]
lagrange_plots = [
    plot_pol_basis_r(p)
    for p in lagrange_polynomials
]

#=======================================
#   Bezier Basis
#=======================================
def bezier_coefficients(i,n):
    x = symbols('x')
    basis_polynomial = sympy.functions.combinatorial.factorials.binomial(n,i) * (x**i) * ((1-x)**(n-i))
    c = basis_polynomial.as_poly().all_coeffs()
    c.reverse()
    return c

def bezier_basis(i, n):
    c = bezier_coefficients(i,n-1)
    return polynomial("e_" + str(i), c, basis_colors[i])


bezier_polynomials = [
    bezier_basis(i, N)
    for i in range(N)
]
bezier_plots = [
    plot_pol_basis_r(p, r_axes_norm)
    for p in bezier_polynomials
]


class p02_0(Scene):
    def construct(self):
        global r_axes, r_axes_norm
        self.add(l_axes, r_axes,Line(UP*5, DOWN*5))
        p_data = [ValueTracker(1),ValueTracker(1), ValueTracker(0), ValueTracker(0)]

        def canonical_basis(i, n):
            c = [0.0] * n
            c[i] = 1
            return polynomial("e_" + str(i),c,basis_colors[i])

        e_polynomials = [
            canonical_basis(i, len(p_data))
            for i in range(len(p_data))
        ]
        e_plots = [
            plot_pol_basis_r(p)
            for p in e_polynomials
        ]

        e0 = polynomial("e_0",[1.0,0.0], basis_colors[0])
        e1 = polynomial("e_1",[0.0,1.0], basis_colors[1])
        d_e0 = Dot(l_axes.c2p(1, 0),color=basis_colors[0])
        d_e1 = Dot(l_axes.c2p(0, 1),color=basis_colors[1])
        t_e0 = polynom_tex(e0).next_to(l_axes,DOWN,buff2,LEFT)
        t_e1 = polynom_tex(e1).next_to(t_e0,DOWN,buff1,LEFT)
        self.add(d_e0,d_e1,t_e0, t_e1,e_plots[0],e_plots[1])

        self.add(*p_data)
        def get_p():
            return polynomial("p", [p.get_value() for p in p_data], BLUE)

        def get_p_p():
            return plot_pol_r(get_p())
        def get_d_p():
            p = get_p()
            return Dot(l_axes.c2p(p.c[0], p.c[1]),color=p.col)
        d_p = always_redraw(get_d_p)
        p_p = always_redraw(get_p_p)
        t_p0 = MathTex("p_0 :=", color=BLUE)
        t_p1 = DecimalNumber(p_data[0].get_value(),1,include_sign=False).next_to(t_p0,RIGHT,buff1)
        t_p1.add_updater(lambda d: d.set_value(p_data[0].get_value()))
        t_p2 = MathTex(e0.n, color=e0.col).next_to(t_p1,RIGHT)
        t_p3 = MathTex('+').next_to(t_p2,RIGHT)
        t_p4 = DecimalNumber(p_data[1].get_value(),1,include_sign=False).next_to(t_p3,RIGHT,buff1)
        t_p4.add_updater(lambda d: d.set_value(p_data[1].get_value()))
        t_p5 = MathTex(e1.n, color=e1.col).next_to(t_p4,RIGHT)
        t_p = VGroup(t_p0,t_p1,t_p2,t_p3,t_p4,t_p5).next_to(t_e1,DOWN,buff2,LEFT)
        self.add(d_p,p_p,t_p)

        self.wait()
        self.play(p_data[0].animate.set_value(3),run_time=2)
        self.play(p_data[1].animate.set_value(0.5),run_time=2)
        self.play(p_data[0].animate.set_value(1),p_data[1].animate.set_value(1),run_time=1)

        self.wait()
        self.play(FadeOut(l_axes),FadeOut(d_e0),FadeOut(d_e1),FadeOut(d_p), FadeOut(t_p),FadeOut(t_e0),FadeOut(t_e1))
        self.play(FadeIn(number_plane))

        e_dots = [
            Dot(color=basis_colors[i]).add_updater(lambda d, i=i: d.move_to(number_plane.c2p(i+1,p_data[i].get_value())))
            for i in range(len(p_data))
        ]
        e_tex = [
            DecimalNumber(p_data[i].get_value(),1,include_sign=True,font_size=DEFAULT_FONT_SIZE*0.8,color=basis_colors[i])
            .move_to(number_plane.c2p(i+1,-3))
            .add_updater(lambda d, i=i: d.set_value(p_data[i].get_value()))
            for i in range(len(p_data))
        ]
        self.play(Create(VGroup(*e_dots)),Create(VGroup(*e_tex)),run_time=2)
        self.play(*[Create(e) for e in e_plots[2:]],run_time=2)
        self.wait(2)

        self.play(p_data[2].animate.set_value( 0.5),run_time=1)
        self.play(p_data[3].animate.set_value( 0.5),run_time=1)
        self.wait(2)


        self.play(
            p_data[1].animate.set_value( 0),
            p_data[2].animate.set_value(0),
            p_data[3].animate.set_value(0),
            *[FadeOut(e) for e in e_plots],
            run_time=1)
        self.wait(2)


        for j in range(len(p_data)):
            self.play(
                *[p_data[i].animate.set_value(lagrange_polynomials[j].c[i]) for i in range(len(p_data))]
            )
            self.wait(2)

        self.play(*[FadeIn(p) for p in lagrange_plots], FadeOut(p_p), *[p.animate.set_value(0) for p in p_data])
        self.wait(2)
        self.play(*[FadeOut(p) for p in lagrange_plots])
        self.play(Transform(r_axes,r_axes_norm))


        self.play(*[FadeIn(p) for p in bezier_plots] )
        self.wait(2)
        self.play(*[FadeOut(p) for p in bezier_plots] )

        self.wait(2)


