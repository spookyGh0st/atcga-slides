import random
import tarfile

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
def lagrange_calc(n,i,t):
    numerator = 1
    denominator = 1
    for m in range(n):
        if m != i:
            numerator *= (t - m)
            denominator *= (i - m)
    return numerator / denominator

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

random.seed(42)
n_points = 8
raw_points = [np.array([1 - float(i) / n_points, random.random()]) for i in range(n_points)]
points = [Dot(p[0] * LEFT * 7 + p[1] * UP * 2 + UP * 2, color=basis_colors[i],z_index=1) for i, p in enumerate(raw_points)]
points_x = [p.get_center()[0] for p in points]
points_y = [p.get_center()[1] for p in points]
def quad_b_spline():
    phi = np.linspace(0,1,n_points)
    b0 = scipy.interpolate.make_interp_spline(phi, np.c_[points_x,points_y])
    b1 = scipy.interpolate.make_interp_spline(phi, np.c_[points_x,points_y])
    b2 = scipy.interpolate.make_interp_spline(phi, np.c_[points_x,points_y])
    b1 = b1.derivative(1)
    b2 = b2.derivative(2)
    return b0,b1,b2




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
        self.play(p_data[0].animate.set_value(3),p_data[1].animate.set_value(0.5),run_time=2); self.next_slide()
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
        self.play(Create(VGroup(*e_dots)),Create(VGroup(*e_tex)),run_time=1)
        self.play(*[Create(e) for e in e_plots[2:]],run_time=1)
        self.next_slide()

        self.play(p_data[2].animate.set_value( 0.5),run_time=0.7)
        self.play(p_data[3].animate.set_value( 0.5),run_time=0.7)
        self.play(p_data[0].animate.set_value( -0.5),run_time=0.7)
        self.play(p_data[1].animate.set_value( -1.0),run_time=0.7)
        self.next_slide()


        self.play(
            p_data[0].animate.set_value( 0),
            p_data[1].animate.set_value( 0),
            p_data[2].animate.set_value(0),
            p_data[3].animate.set_value(0),
            *[FadeOut(e) for e in e_plots],
            run_time=1)
        self.wait(0.1)
        self.next_slide()

        for j in range(len(p_data)):
            self.play(
                *[p_data[i].animate.set_value(lagrange_polynomials[j].c[i]) for i in range(len(p_data))]
            )
            self.wait(1)

        lagr_tex = MathTex(r'L_i^n(t) = ',r'{{(t-t_0) \ldots (t-t_{i-1})(t-t_{i+1}) \ldots (t-t_n)} \over',
                           r'(t_i-t_0) \ldots (t_i-t_{i-1})(t_i-t_{i+1}) \ldots (t_i-t_n)}}',font_size=DEFAULT_FONT_SIZE/2).to_corner(DL)
        lagr_tex2 = MathTex(r'p(t) = \sum_{i=0}^n P_i L_i^n(t)',font_size=DEFAULT_FONT_SIZE/2).to_corner(DL).next_to(lagr_tex,UP,buff2,LEFT)
        self.play(*[FadeIn(p) for p in lagrange_plots],
                  *[FadeOut(i) for i in [p_p,*e_dots,number_plane,*e_tex]],
                  *[p.animate.set_value(0) for p in p_data],Create(lagr_tex), Create(lagr_tex2))
        self.next_slide()

        t = ValueTracker(0); self.add(t)
        t_li = always_redraw(lambda: Line(r_axes.c2p(t.get_value(),0),r_axes.c2p(t.get_value(),1),color=RED))
        t_dec = (DecimalNumber(t.get_value(),2)
                 .add_updater(lambda d: d.next_to(t_li,DOWN).set_value(t.get_value()),call_updater=True))
        self.play(*[FadeIn(p) for p in points[:4]], Create(t_li),Create(t_dec),run_time=1.0)


        def lagrange_curve(t:float) ->np.ndarray:
            return np.sum([lagrange_calc(4, i, t) * points[i].get_center() for i in range(4)],axis=0)
        lag_c = always_redraw(lambda: ParametricFunction(lagrange_curve, t_range = np.array([0, 3]), fill_opacity=0).set_color(RED))
        self.play(Create(lag_c),t.animate.set_value(3),run_time=4); self.next_slide()
        c1 = always_redraw(lambda: Circle(radius=0.2,color=RED).move_to(points[1].get_center()))
        c2 = always_redraw(lambda: Circle(radius=0.2,color=RED).move_to(points[2].get_center()))
        self.play(Create(c1),Create(c2))
        self.play(points[1].animate.shift(UP),points[2].animate.shift(DOWN*3)); self.next_slide()
        self.play(points[1].animate.shift(DOWN),points[2].animate.shift(UP*3))
        self.next_slide()

        self.play(*[FadeOut(p) for p in lagrange_plots], FadeOut(lagr_tex),FadeOut(lagr_tex2), *[FadeOut(i) for i in [lag_c, c1,c2, t_li,t_dec]], run_time=1.0)
        self.play(Transform(r_axes,r_axes_norm))
        self.next_slide()



class p02_1(Slide):

    def construct(self):
        global r_axes, r_axes_norm
        n_sec1 = 4
        self.add(r_axes_norm,Line(UP*5, DOWN*5))
        self.add(*points[:4])
        self.add(*bezier_plots)
        bez_bas_tex = MathTex(r'B_i^n(t) = \binom{n}{i} t^i (1-t)^{n-1}').to_corner(DL)
        bez_bas_tex2 = MathTex(r'p(t) = \sum^n_{i=0} P_i B^n_i(t)').next_to(bez_bas_tex,UP,buff2,LEFT)
        self.play(*[Create(p) for p in bezier_plots] ,Create(bez_bas_tex))
        self.play(*[Create(p) for p in points[4:]],Create(bez_bas_tex2), run_time=1.0)


        def bez_curve_1(t:float) ->np.ndarray:
            return np.sum([besier_calc(i, n_sec1-1, t) * points[i].get_center() for i in range(n_sec1)],axis=0)
        def bez_curve_2(t:float) ->np.ndarray:
            return np.sum([besier_calc(i, n_sec1-1, t) * points[i+n_sec1].get_center() for i in range(n_sec1)],axis=0)

        bez_1 = ParametricFunction(bez_curve_1, t_range = np.array([0, 1]), fill_opacity=0).set_color(RED)
        t = ValueTracker(0); self.add(t)
        t_li = always_redraw(lambda: Line(r_axes_norm.c2p(t.get_value(),0),r_axes_norm.c2p(t.get_value(),1),color=RED))
        t_dec = (DecimalNumber(t.get_value(),2)
                 .add_updater(lambda d: d.next_to(t_li,DOWN).set_value(t.get_value()),call_updater=True))
        self.next_slide()
        self.play(Create(t_li),Create(t_dec),run_time=0.5)
        self.play(Create(bez_1),t.animate.set_value(1),run_time=5)
        self.next_slide()
        conv_hull1 = Polygon(*[p.get_center() for p in [points[0],points[1],points[3]]],fill_opacity=0.8,z_index=-1, fill_color=ORANGE)
        conv_hull2 = Polygon(*[r_axes_norm.c2p(0,0),r_axes_norm.c2p(1,0), r_axes_norm.c2p(1,1), r_axes_norm.c2p(0,1) ],fill_opacity=0.8,z_index=-1, fill_color=ORANGE)
        self.play(Create(conv_hull1),Create(conv_hull2)); self.wait(0.1)
        self.next_slide()
        self.play(Uncreate(conv_hull1),Uncreate(conv_hull2))

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
        ),FadeOut(bez_bas_tex),FadeOut(bez_bas_tex2),run_time=2)
        self.next_slide()
        ax = Axes(
            x_range=[-0.5, 2.5, 1],
            y_range=[-0, 1.01, 1],
            x_length=6,
            y_length=2, tips=False
        ).to_corner(LEFT+DOWN)
        T = np.array([0,0,0,0,1,1,1,2,2,2,2])
        T_tex = MathTex('T = ['+",".join(str(s) for s in T) + ']',font_size=0.8*DEFAULT_FONT_SIZE).next_to(ax, UP)
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
        self.play(Create(VGroup(*norm_b_plots)),run_time = 5)

        self.next_slide()
        self.play(*[FadeOut(p) for p in norm_b_plots],FadeOut(T_tex))


        T = np.array([0,0,0,0,1,2,3,4,5,5,5,5]) * (2/5)
        T_tex = MathTex('T = ['+','.join(["{:.1f}".format(s) for s in T])+']',font_size=DEFAULT_FONT_SIZE*0.4).next_to(ax, UP)
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

class p02_2(Slide):
    def construct(self):
        ax = Axes(
            x_range=[-0.1, 1.1, 1],
            y_range=[-0, 4.01, 1],
            x_length=5,
            y_length=5, tips=False
        ).to_edge(RIGHT)
        self.add(ax,Line(UP*5, DOWN*5))
        self.add(*points)
        b0,b1,b2 = quad_b_spline()
        N = 4*32
        phi_new = np.linspace(0,1,4*32)
        x,y = b0(phi_new).T; c = np.column_stack((x,y,np.zeros(N)))
        x,y = b1(phi_new).T; dc = np.column_stack((x,y,np.zeros(N)))
        x,y = b2(phi_new).T; ddc = np.column_stack((x,y,np.zeros(N)))
        kur = np.divide(np.linalg.norm(np.cross(ddc, dc),axis=1), np.power(np.linalg.norm(dc,axis=1),3))

        def c_kur(t:float):
            return kur[int(t*(N-1))]
        def c_c(t:float):
            return c[int(t*(N-1))]
        def c_dc(t: float):
            return dc[int(t*(N-1))]
        def c_ddc(t):
            return ddc[int(t*(N-1))]
        def c_T(t:float):
            return c_dc(t)/np.linalg.norm(c_dc(t))
        def c_N(t:float):
            Td = np.cross(c_dc(t), np.cross(c_ddc(t), c_dc(t)))
            return Td / np.linalg.norm(Td)
        def c_dc_norm(t: float):
            return dc[int(t*(N-1))] / 10 # np.linalg.norm(dc[int(t*(N-1))])
        def c_ddc_norm(t):
            return ddc[int(t*(N-1))] / 200 #np.linalg.norm(ddc[int(t*(N-1))])
        v_t = ValueTracker(0); self.add(v_t)
        man_dc = always_redraw(lambda : Arrow(c_c(v_t.get_value()),c_c(v_t.get_value())+c_dc_norm(v_t.get_value()),buff=0,color=MAROON))
        man_ddc = always_redraw(lambda: Arrow(c_c(v_t.get_value())+c_dc_norm(v_t.get_value()),c_c(v_t.get_value())+c_dc_norm(v_t.get_value())+c_ddc_norm(v_t.get_value()),buff=0,color=RED_B))
        man_N = always_redraw(lambda : Arrow(c_c(v_t.get_value()),c_c(v_t.get_value())+c_N(v_t.get_value()),buff=0,color=GREEN))
        man_c = ParametricFunction(lambda t: c[int(t*(N-1))],np.array([0,1]),color=RED)
        man_kur = ax.plot(c_kur,np.array([0,1]),color=GOLD,z_index=1)

        t_li = always_redraw(lambda: Line(ax.c2p(v_t.get_value(),0),ax.c2p(v_t.get_value(),1),color=RED))
        t_dec = (DecimalNumber(v_t.get_value(),2)
                 .add_updater(lambda d: d.next_to(t_li,DOWN).set_value(v_t.get_value()),call_updater=True))
        circ = always_redraw(lambda: Circle(z_index = 2, color=ORANGE, radius = 1./c_kur(v_t.get_value())).move_to(c_c(v_t.get_value())+1./c_kur(v_t.get_value())*c_N(v_t.get_value()))) #+c_ddc(v_t.get_value())/c_kur(v_t.get_value()))

        self.add(man_dc,man_ddc,man_N,man_kur,t_li,t_dec,circ)
        self.play(Create(man_c),v_t.animate.set_value(1),run_time=10)

