import random

import numpy as np
import scipy.stats
from manim import *
from manim_slides.slide import Slide
from dataclasses import dataclass
from sympy import symbols
import sympy as sympy



colormap = np.array([
     [ 0.0503820534705988, 0.0298017364997418  , 0.527975101049518 ,1],
     [ 0.134263095956425 , 0.0221290414640333  , 0.564086943874442 ,1],
     [ 0.19658429522455  , 0.0181487230853725  , 0.591699907010603 ,1],
     [ 0.252508903744709 , 0.0140597298370623  , 0.614596052572696 ,1],
     [ 0.305597246846751 , 0.00896293218281256 , 0.633493656287653 ,1],
     [ 0.357161342424505 , 0.00372244628755486 , 0.64799537276208  ,1],
     [ 0.40766452865892  , 0.000635421871269555, 0.657244390906515 ,1],
     [ 0.45713493296072  , 0.00321996649814875 , 0.660291088653498 ,1],
     [ 0.505338779334359 , 0.0158799603613645  , 0.656373803832435 ,1],
     [ 0.551894068905485 , 0.0432673509661537  , 0.645217958234789 ,1],
     [ 0.596370855317254 , 0.0783105461525145  , 0.627252928317216 ,1],
     [ 0.638408558026694 , 0.114225740031185   , 0.603686083070476 ,1],
     [ 0.677817333874352 , 0.150564237726243   , 0.576205185917591 ,1],
     [ 0.714604238385044 , 0.187011213613401   , 0.546574897373828 ,1],
     [ 0.748939980627268 , 0.223423831835724   , 0.516239938838344 ,1],
     [ 0.781069149977098 , 0.259789608780162   , 0.486138795941628 ,1],
     [ 0.811236242283869 , 0.296210417771779   , 0.45670663787859  ,1],
     [ 0.839614821321557 , 0.33285651624084    , 0.428012933217697 ,1],
     [ 0.866280201095155 , 0.369951912397464   , 0.399908030028535 ,1],
     [ 0.891185920886601 , 0.407746777797117   , 0.37215683516806  ,1],
     [ 0.914179886198195 , 0.446494219266261   , 0.344520445409926 ,1],
     [ 0.935012277718101 , 0.486438292121575   , 0.316826252519024 ,1],
     [ 0.953361035955715 , 0.527793036024391   , 0.288993337404004 ,1],
     [ 0.968852021999771 , 0.570730380904922   , 0.261068305515935 ,1],
     [ 0.981089981012577 , 0.615358169581108   , 0.233246532554544 ,1],
     [ 0.989650252591629 , 0.661747758025894   , 0.206018553736301 ,1],
     [ 0.994066749767429 , 0.709927390700005   , 0.180482004740676 ,1],
     [ 0.993816357899574 , 0.75990211771125    , 0.1588815644222   ,1],
     [ 0.988319905382877 , 0.811642117364466   , 0.14503890140546  ,1],
     [ 0.977050900222136 , 0.865041805460105   , 0.143105142180541 ,1],
     [ 0.959823007609692 , 0.919885356968148   , 0.151329147688882 ,1],
     [ 0.940015127878274 , 0.975155785620538   , 0.131325887773911 ,1]
])*255

buff1 = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
buff2 = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2

lang = r'\left\langle'
rang = r'\right\rangle'


def gen_omega_tex():
    return r'{ \Omega }'
def gen_boundary_tex():
    return r'{\partial \Omega}'
def gen_u0_tex():
    return r'u_0'
def gen_u_tex():
    return r'{ u }'

def setColors(tex: MathTex):
    tex.set_color_by_tex(gen_boundary_tex(), PURPLE)
    tex.set_color_by_tex(gen_omega_tex(), ORANGE)
    tex.set_color_by_tex(gen_u0_tex(), TEAL)
    tex.set_color_by_tex(gen_u_tex(), MAROON)
    return tex

class p03_0(Slide):
    def ns(self):
        self.next_slide();
        # self.wait(1)


    def construct(self):
        radius = 1.2
        open_c = Circle(fill_color=ORANGE,fill_opacity=1,stroke_opacity=0,radius=radius).to_corner(UP+LEFT)
        closed_c = Circle(stroke_color=PURPLE,fill_opacity=0,stroke_opacity=1,radius=radius).to_corner(UP+LEFT)
        open_set_tex = setColors(MathTex(' \\text{Open Set }\,', gen_omega_tex()).next_to(open_c, RIGHT).shift(UP*0.5))
        closed_set_tex = setColors(MathTex('\\text{boundary }\,', gen_boundary_tex()).next_to(open_c, RIGHT).shift(DOWN*0.5))
        self.play(Create(open_c),Write(open_set_tex)); self.ns()
        self.play(Create(closed_c), Write(closed_set_tex)); self.ns()
        u0_tex = setColors(MathTex(r'\text{given }',gen_u0_tex(), ': ', gen_boundary_tex(), r'\to \mathbf{R}')).next_to(open_c,DOWN,buff2,LEFT)
        u_tex1 = setColors(MathTex(r'\text{find }\,\,',gen_u_tex(), ': ', gen_omega_tex(), r'\to \mathbf{R} ')).next_to(u0_tex,DOWN,buff2,LEFT)
        u_tex2 = MathTex(r'\text{ as "flat" as possible}').next_to(u_tex1,DOWN,buff1,LEFT)
        self.play(FadeIn(u0_tex)); self.ns()
        self.play(FadeIn(u_tex1))
        self.play(FadeIn(u_tex2))
        self.ns()

        def gradient_ur_f(pos:np.ndarray) -> np.ndarray:
            if (np.linalg.norm(pos)>=radius): return np.array([0,0,0])
            p = np.array([1,pos[1]*pos[0]])
            return 2*(p / np.linalg.norm(p))
        def gradient_dr_f(pos:np.ndarray) -> np.ndarray:
            if (np.linalg.norm(pos)>=radius): return np.array([0,0,0])
            return np.array([1.0,0.,0.])
        gradient_range = [-radius,radius]
        closed_c_ur = Circle(stroke_color=PURPLE,fill_opacity=0,stroke_opacity=1,radius=radius).to_corner(UR)
        gradient_ur = ArrowVectorField(gradient_ur_f,x_range=gradient_range,y_range=gradient_range).move_to(closed_c_ur.get_center()+UR*radius/6)
        closed_c_dr = Circle(stroke_color=PURPLE,fill_opacity=0,stroke_opacity=1,radius=radius).to_corner(DR)
        gradient_dr = ArrowVectorField(gradient_dr_f,x_range=gradient_range,y_range=gradient_range).move_to(closed_c_dr.get_center()+UR*radius/2)
        grad_l = Line(radius*LEFT,radius*RIGHT).to_edge(RIGHT)
        grad_l_d1 = DecimalNumber(0.0).next_to(grad_l,DOWN,buff1,LEFT)
        grad_l_d2 = DecimalNumber(1.0).next_to(grad_l,DOWN,buff1,RIGHT)
        self.play(*[Create(x) for x in [closed_c_ur,closed_c_dr,grad_l,grad_l_d1,grad_l_d2]])
        self.ns()

        self.play(Create(gradient_ur),Create(gradient_dr))
        self.ns()

        min1 = setColors(MathTex(r'\min_',gen_u_tex(), r'\int_',gen_omega_tex(),r'||(\nabla u) (x) ||^2_2 \, dx')).next_to(u_tex2,DOWN,buff2,LEFT)
        min2 = setColors(MathTex(r'\text{subject to }',gen_u_tex(), r'\vert_',gen_boundary_tex(),r' =',gen_u0_tex())).next_to(min1,DOWN,buff1,LEFT)
        self.play(FadeIn(min1),FadeIn(min2))
        self.ns()

        c1 = Circle(radius=0.4*radius).move_to(closed_c_ur.get_center()+LEFT*radius/2)
        c2 = Circle(radius=0.4*radius).move_to(closed_c_dr.get_center()+LEFT*radius/2)
        self.play(Create(c1),Create(c2))
        self.ns()

        min3 = setColors(MathTex(r'\Leftrightarrow \text{ solve }',r'\Delta',gen_u_tex(),r'\vert_',gen_omega_tex(),'=0')).next_to(u_tex2,DOWN,buff2,LEFT)
        min4 = setColors(MathTex(r'\text{ with }',gen_u_tex(), r'\vert_',gen_boundary_tex(),r' =',gen_u0_tex())).next_to(min3,DOWN,buff1,LEFT)
        self.play(Transform(min1,min3),Transform(min2,min4))
        self.ns()

        min5 = setColors(MathTex(r'\Leftrightarrow \text{ solve }',r'\Delta',gen_u_tex(),r'\vert_',gen_omega_tex(),'= f')).next_to(u_tex2,DOWN,buff2,LEFT)
        self.play(Transform(min1,min5))
        self.ns()

        min6 = setColors(MathTex(r'\Rightarrow \text{ solve }',r'\int_',gen_omega_tex(),r'h(x) (\Delta',gen_u_tex(),r') (x) \, dx = \int_',gen_omega_tex(),'h(x)f(x) dx')).next_to(u_tex2,DOWN,buff2,LEFT)
        self.play(Transform(min1,min6),min2.animate.next_to(min6,DOWN,buff1,LEFT))
        self.ns()


        min7 =  setColors(MathTex(r'\Rightarrow \text{ solve } \int_', gen_omega_tex(), lang, r'(\nabla', gen_u_tex(), r')(x)\vert', r'(\nabla h)(x)', rang,r' \, dx = \int_',gen_omega_tex(),'h(x)f(x) \, dx')).next_to(u_tex2,DOWN,buff2,LEFT)
        self.play(Transform(min1,min7),min2.animate.next_to(min7,DOWN,buff1,LEFT))
        self.ns()

        min7 =  setColors(MathTex(r'\Leftrightarrow \text{solve }' ,lang, r'h \vert', gen_u_tex(), rang,r'_\Delta',r'= \int_',gen_omega_tex(),'h(x)f(x) \, dx')).next_to(u_tex2,DOWN,buff2,LEFT)
        self.play(Transform(min1,min7),min2.animate.next_to(min7,DOWN,buff1,LEFT))
        self.ns()

        self.play(*[FadeOut(x) for x in [open_c,closed_c,open_set_tex,closed_set_tex,closed_c_ur,closed_c_dr,gradient_dr,gradient_ur,u0_tex,u_tex1,u_tex2,grad_l,grad_l_d1,grad_l_d2]])
        min8 =  setColors(MathTex(r'\text{solve }' ,lang, r'h \vert', gen_u_tex(), rang,r'_\Delta',r'= \int_',gen_omega_tex(),'h(x)f(x) \, dx')).to_corner(UL)
        self.play(Transform(min1,min8),min2.animate.next_to(min8,DOWN,buff1,LEFT))
        self.ns()


class p03_1(Slide):
    def ns(self):
        #self.next_slide();
        self.wait(1)
    def construct(self):
        min1 =  setColors(MathTex(r'\text{solve }' ,lang, r'h \vert', gen_u_tex(), rang,r'_\Delta',r'= \int_',gen_omega_tex(),'h(x)f(x) \, dx')).to_corner(UL)
        min2 = setColors(MathTex(r'\text{subject to }',gen_u_tex(), r'\vert_',gen_boundary_tex(),r' =',gen_u0_tex())).next_to(min1,DOWN,buff1,LEFT)
        self.add(min1,min2)
        self.ns()





