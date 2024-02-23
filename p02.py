import random

import numpy as np
import scipy.stats
from manim import *
from manim_slides.slide import Slide
from dataclasses import dataclass
from sympy import symbols
import sympy as sympy

N = 4
basis_colors = [PURPLE, ORANGE, GREEN, MAROON, GOLD, TEAL, GRAY, BLUE, ORANGE, GREEN, DARK_BROWN ]

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
def besier_calc(i,n,t):
    return scipy.special.binom(n,i)*(t**i)*((1-t)**(n-i))

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

### Basis_splines
def norm_b_splines(r:int,i:int,t:float,T:np.ndarray)->float:
    if (r == 0):
        if (T[i] <= t and t < T[i+1]): # TODO
            return 1
        else:
            return 0
    else:
        if (T[i+r]-T[i] == 0):
            n1 = 0
        else:
            n1 = (t - T[i])/ (T[i+r]-T[i]) * norm_b_splines(r-1,i,t,T)
        if ((T[i+1+r] - T[i+1]) == 0 ):
            n2 = 0
        else:
            n2 = (T[i+1+r] - t) / (T[i+1+r] - T[i+1]) * norm_b_splines(r-1,i+1,t,T)
        return n1 + n2


class p02_0(Slide):
    def construct(self):
        global r_axes, r_axes_norm
        self.play(*[Create(i) for i in [l_axes, r_axes,Line(UP*5, DOWN*5)]]);
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
        self.play(*[Create(i) for i in [d_e0,d_e1,t_e0, t_e1,e_plots[0],e_plots[1]]]); self.next_slide()

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
        self.play(*[Create(i) for i in [d_p,p_p,t_p]])
        self.next_slide()

        self.wait()
        self.play(p_data[0].animate.set_value(3),run_time=2); self.next_slide()
        self.play(p_data[1].animate.set_value(0.5),run_time=2); self.next_slide()
        self.play(p_data[0].animate.set_value(1),p_data[1].animate.set_value(1),run_time=1); self.next_slide()

        self.play(FadeOut(l_axes),FadeOut(d_e0),FadeOut(d_e1),FadeOut(d_p), FadeOut(t_p),FadeOut(t_e0),FadeOut(t_e1))
        self.play(FadeIn(number_plane)); self.next_slide()

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
        self.next_slide()

        self.play(p_data[2].animate.set_value( 0.5),run_time=1)
        self.play(p_data[3].animate.set_value( 0.5),run_time=1)
        self.next_slide()


        self.play(
            p_data[1].animate.set_value( 0),
            p_data[2].animate.set_value(0),
            p_data[3].animate.set_value(0),
            *[FadeOut(e) for e in e_plots],
            run_time=1)
        self.next_slide()


        for j in range(len(p_data)):
            self.play(
                *[p_data[i].animate.set_value(lagrange_polynomials[j].c[i]) for i in range(len(p_data))]
            )
            self.wait(1)

        self.play(*[FadeIn(p) for p in lagrange_plots], FadeOut(p_p), *[p.animate.set_value(0) for p in p_data])
        self.next_slide()
        self.play(*[FadeOut(p) for p in lagrange_plots])
        self.play(Transform(r_axes,r_axes_norm))


        self.play(*[FadeIn(p) for p in bezier_plots] )
        self.next_slide()



class p02_1(Slide):

    def construct(self):
        global r_axes, r_axes_norm
        random.seed(42)
        n_points = 8
        n_sec1 = 4
        self.add(r_axes_norm,Line(UP*5, DOWN*5))
        self.add(*bezier_plots)
        raw_points = [ np.array([1-float(i)/n_points,random.random()]) for i in range(n_points) ]
        points = [ Point(p[0]*LEFT*7+p[1]*UP*2+UP*2,color=basis_colors[i]) for i,p in enumerate(raw_points)]
        self.play(*[FadeIn(p) for p in points], run_time=1.0)

        def bez_curve_1(t:float) ->np.ndarray:
            return np.sum([besier_calc(i, n_sec1-1, t) * points[i].get_center() for i in range(n_sec1)],axis=0)
        def bez_curve_2(t:float) ->np.ndarray:
            return np.sum([besier_calc(i, n_sec1-1, t) * points[i+n_sec1].get_center() for i in range(n_sec1)],axis=0)

        bez_1 = ParametricFunction(bez_curve_1, t_range = np.array([0, 1]), fill_opacity=0).set_color(RED)
        t = ValueTracker(0); self.add(t)
        t_li = always_redraw(lambda: Line(r_axes_norm.c2p(t.get_value(),0),r_axes_norm.c2p(t.get_value(),1),color=RED))
        t_dec = (DecimalNumber(t.get_value(),2)
                 .add_updater(lambda d: d.next_to(t_li,DOWN).set_value(t.get_value()),call_updater=True))
        self.play(Create(t_li),Create(t_dec),run_time=0.5)
        self.play(Create(bez_1),t.animate.set_value(1),run_time=5)
        self.next_slide()

        t.set_value(0)
        for i in range(len(bezier_plots)):
            bezier_plots[i].set_color(basis_colors[i+len(bezier_plots)])
        bez_2 = always_redraw(lambda: ParametricFunction(bez_curve_2, t_range = np.array([0, t.get_value()]), fill_opacity=0).set_color(RED))
        self.play(Create(bez_2)); self.play(t.animate.set_value(1),run_time=2)

        self.next_slide()
        self.play(FadeOut(t_li),FadeOut(t_dec), points[4].animate.move_to(points[3].get_center()),run_time=2)

        self.next_slide()
        self.play(points[5].animate.move_to(
            points[3].get_center()-(points[2].get_center()-points[3].get_center())
        ),run_time=2)
        self.next_slide()
        ax = Axes(
            x_range=[-0.5, 2.5, 1],
            y_range=[-0, 1.01, 1],
            x_length=6,
            y_length=2, tips=False
        ).to_corner(LEFT+DOWN)
        T = np.array([0,0,0,0,1,1,1,2,2,2,2])
        T_tex = MathTex('T = '+",".join(str(s) for s in T),font_size=0.8*DEFAULT_FONT_SIZE).next_to(ax, UP)
        self.play(Create(ax),Write(T_tex))
        self.next_slide()
        size = len(T)-1
        degree = 3
        norm_b_plots = [
            ax.plot(
                lambda t: norm_b_splines(degree,i,t,T),
                use_smoothing=False,
                color=basis_colors[i],
                discontinuities = [0,1,2],
            )
            for i in range(size-degree)
        ]
        self.play(Create(VGroup(*norm_b_plots)),run_time = 5)

        self.next_slide()
        self.play(*[FadeOut(p) for p in norm_b_plots],FadeOut(T_tex))


        T = np.array([0,0,0,0,1,2,3,4,5,5,5,5]) * (2/5)
        T_tex = MathTex('T = '+','.join(["{:.1f}".format(s) for s in T]),font_size=DEFAULT_FONT_SIZE*0.4).next_to(ax, UP)
        size = len(T)-1
        degree = 3
        norm_b_plots = [
            ax.plot(
                lambda t: norm_b_splines(degree,i,t,T),
                use_smoothing=False,
                color=basis_colors[i],
                discontinuities = [0,1,2],
            )
            for i in range(size-degree)
        ]
        self.play(Create(ax),Write(T_tex))
        self.play(Create(VGroup(*norm_b_plots)),run_time = 1)
        self.next_slide()
        self.play(points[4].animate.move_to(points[4].get_center()+UP),run_time=1)
        self.next_slide()

        def b_spline_curve(t:float) ->np.ndarray:
            return np.sum([norm_b_splines(degree,i,t,T) * points[i].get_center() for i in range(len(points))],axis=0)

        b_spline_1 = ParametricFunction(b_spline_curve, t_range = np.array([0.000001, 1.9999]), fill_opacity=0).set_color(RED_A)
        t_new = ValueTracker(0); self.add(t_new)
        t_li = always_redraw(lambda: Line(ax.c2p(t_new.get_value(),0),ax.c2p(t_new.get_value(),1),color=RED))
        t_dec = (DecimalNumber(t_new.get_value(),2)
                 .add_updater(lambda d: d.next_to(t_li,DOWN).set_value(t_new.get_value()),call_updater=True))
        self.play(Create(t_li),Create(t_dec),run_time=0.5)
        self.next_slide()
        self.play(t_new.animate.set_value(2),Create(b_spline_1),run_time=5)
        self.next_slide()
