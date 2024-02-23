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
def gen_fb_tex():
    return r'{ \Phi }'

def gen_S():
    return r'\mathbf{S}'
def gen_a():
    return r'\mathbf{a}'
def gen_b():
    return r'\mathbf{b}'

def setColors(tex: MathTex):
    tex.set_color_by_tex(gen_boundary_tex(), PURPLE)
    tex.set_color_by_tex(gen_omega_tex(), ORANGE)
    tex.set_color_by_tex(gen_u0_tex(), TEAL)
    tex.set_color_by_tex(gen_u_tex(), MAROON)
    tex.set_color_by_tex(gen_fb_tex(), GOLD_A)
    tex.set_color_by_tex(gen_S(), GOLD_A)
    tex.set_color_by_tex(gen_a(), MAROON)
    tex.set_color_by_tex(gen_b(), TEAL)
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
        u0_tex = setColors(MathTex(r'\text{given }',gen_u0_tex(), ': ', gen_boundary_tex(), r'\to \mathbb{R}')).next_to(open_c,DOWN,buff2,LEFT)
        u_tex1 = setColors(MathTex(r'\text{find }\,\,',gen_u_tex(), ': ', gen_omega_tex(), r'\to \mathbb{R} ')).next_to(u0_tex,DOWN,buff2,LEFT)
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


        min7 =  setColors(MathTex(r'\Rightarrow \text{ solve } -\int_', gen_omega_tex(), lang, r'(\nabla', gen_u_tex(), r')(x)\vert', r'(\nabla h)(x)', rang,r' \, dx = \int_',gen_omega_tex(),'h(x)f(x) \, dx')).next_to(u_tex2,DOWN,buff2,LEFT)
        self.play(Transform(min1,min7),min2.animate.next_to(min7,DOWN,buff1,LEFT))
        self.ns()

        min7 =  setColors(MathTex(r'\Leftrightarrow \text{solve } -' ,lang, r'h \vert', gen_u_tex(), rang,r'_\Delta',r'= \int_',gen_omega_tex(),'h(x)f(x) \, dx')).next_to(u_tex2,DOWN,buff2,LEFT)
        self.play(Transform(min1,min7),min2.animate.next_to(min7,DOWN,buff1,LEFT))
        self.ns()

        self.play(*[FadeOut(x) for x in [open_c,closed_c,open_set_tex,closed_set_tex,closed_c_ur,closed_c_dr,gradient_dr,gradient_ur,u0_tex,u_tex1,u_tex2,grad_l,grad_l_d1,grad_l_d2]])
        min8 =  setColors(MathTex(r'\text{solve }-' ,lang, r'h \vert', gen_u_tex(), rang,r'_\Delta',r'= \int_',gen_omega_tex(),'h(x)f(x) \, dx')).to_corner(UL)
        self.play(Transform(min1,min8),min2.animate.next_to(min8,DOWN,buff1,LEFT))
        self.ns()


vertices_c = 4
vertices_x = np.linspace(0.,1.1,num=vertices_c)
vertices_y = np.linspace(0.,1.1,num=vertices_c)
vertices = np.vstack((vertices_x.flatten(), vertices_y.flatten())).T

axes_UR = Axes(x_range=[-0.1,1.1],y_range=[-0.1,1.1],x_length=4,y_length=4).to_corner(UR)
def ap(a: np.ndarray)->np.ndarray:
    return axes_UR.c2p(a[0],a[1])
dots = [
    Dot(axes_UR.c2p(vertices_x[i],vertices_y[j]),color=(PURPLE if i == 0 or i == vertices_c-1 or j == 0 or j == vertices_c-1 else ORANGE),z_index=1)
    for j in range(len(vertices_y)) for i in range(len(vertices_x))
]
simplices = []
for i in range(vertices_c - 1):
    for j in range(vertices_c - 1):
        # Define vertices of each triangle in counterclockwise order
        v1 = i * vertices_c + j
        v2 = v1 + 1
        v3 = v1 + vertices_c
        simplices.append([v1, v2, v3])

        v1 = v2
        v2 = v3 + 1
        v3 = v3
        simplices.append([v1, v2, v3])
simplices = np.array(simplices,dtype=int)
m_simplices = [
    VGroup(
        Line(dots[s[0]].get_center(), dots[s[1]].get_center()),
        Line(dots[s[1]].get_center(), dots[s[2]].get_center()),
        Line(dots[s[2]].get_center(), dots[s[0]].get_center()),
    )
    for s in simplices
]
fin_basis = [Polygon(
    dots[1 + i + j * vertices_c].get_center(),
    dots[2 + i + j * vertices_c].get_center(),
    dots[6 + i + j * vertices_c].get_center(),
    dots[9 + i + j * vertices_c].get_center(),
    dots[8 + i + j * vertices_c].get_center(),
    dots[4 + i + j * vertices_c].get_center(),
    fill_opacity=0.5, stroke_opacity = 0.5,
    fill_color=GOLD_A, z_index=-1
) for i in range(2) for j in range(2)]
fin_basis_tex = [
    setColors(MathTex(gen_fb_tex(), '_{', str(i), '}').next_to(axes_UR, DOWN)) for i in [5, 6, 9, 10]
]

class p03_1(Slide):
    def ns(self):
        self.next_slide();
        #self.wait(0.1)
    def construct(self):
        min1 =  setColors(MathTex(r'\text{solve } -' ,lang, r'h \vert', gen_u_tex(), rang,r'_\Delta',r'= \int_',gen_omega_tex(),'h(x)f(x) \, dx')).to_corner(UL)
        min2 = setColors(MathTex(r'\text{subject to }',gen_u_tex(), r'\vert_',gen_boundary_tex(),r' =',gen_u0_tex())).next_to(min1,DOWN,buff1,LEFT)
        self.add(min1,min2); self.wait()
        self.play(*[Create(i) for i in dots],*[Create(i) for i in m_simplices])
        self.ns()
        self.play(Create(fin_basis[0]),Create(fin_basis_tex[0])); self.ns()
        self.remove(fin_basis[0],fin_basis_tex[0]); self.add(fin_basis[1],fin_basis_tex[1])
        self.wait(); self.remove(fin_basis[1],fin_basis_tex[1]); self.add(fin_basis[2],fin_basis_tex[2])
        self.wait(); self.remove(fin_basis[2],fin_basis_tex[2]);
        self.add(fin_basis[3],fin_basis_tex[3]); self.ns()

        ud1 = MathTex(r'\text{Rewrite Function as}').next_to(min2,DOWN,buff2,LEFT)
        ud2 = setColors(MathTex(gen_u_tex(),r'(x) = \sum_{j \in I} a_j ',gen_fb_tex(),r'_j(x) \, \text{ for some } a \in \mathbb{R}^n').next_to(ud1,DOWN,buff1,LEFT))
        ud3 = setColors(MathTex(r'h(x) = \sum_{j \in I} b_j ',gen_fb_tex(),r'_j(x) \, \text{ for some } b \in \mathbb{R}^n \, \text{ , } \, b \vert_',gen_boundary_tex(),r'=0').next_to(ud2,DOWN,buff1,LEFT))
        self.play(FadeIn(ud1),FadeIn(ud2)); self.ns()
        self.play(FadeIn(ud3)); self.ns()

        min3 =  setColors(MathTex(r' -' ,lang, gen_fb_tex(), r'_i \vert', r'\sum_{j \in I} a_j',gen_fb_tex(),'_j', rang,r'_\Delta',r'= \int_',gen_omega_tex(),gen_fb_tex(),r'_i \left(\sum_{j \in i} b_j',gen_fb_tex(),r'(x)\right) \, dx')).to_corner(UL)
        min4 = setColors(MathTex(r'\text{for all } i \in I_{',gen_omega_tex(),r'\setminus',gen_boundary_tex(),'}')).next_to(min3,DOWN,2*buff2,LEFT)
        self.play(Transform(min1,min3), Transform(min2,min4),FadeOut(ud1),FadeOut(ud2),FadeOut(ud3))
        self.ns()
        min5 =  setColors(
            MathTex(r'\sum_{j \in I \vert_{',gen_omega_tex(),r'\setminus',gen_boundary_tex(),'} }', lang, gen_fb_tex(), r'_i \vert', gen_fb_tex(),'_j',rang,r'_\Delta',
                    r'= \sum_{j \in I} b_j \int_',gen_omega_tex(),gen_fb_tex(),'_i (x)', gen_fb_tex(), r'_j(x) \, dx',
                    r'+ \sum_{j \in I \vert_', gen_boundary_tex(),r'} a_j', lang, gen_fb_tex(), r'_i \vert', gen_fb_tex(),'_j',rang,r'_\Delta',
                    font_size=0.9*DEFAULT_FONT_SIZE)).next_to(min4,DOWN,3*buff2,LEFT)
        self.play(FadeIn(min5));self.ns()

        min6 =  setColors( MathTex(r'\text{solve } ',gen_S(),gen_a(),r' = ',gen_b())).to_corner(UL)

        self.play(Transform(min1,min6),Transform(min5,min6),Transform(min2,min6)); self.ns()


class p03_2(Slide):
    def ns(self):
        self.next_slide();
        # self.wait(0.1)
    def construct(self):
        min1 =  setColors( MathTex(r'\text{solve } ',gen_S(),gen_a(),r' = ',gen_b())).to_corner(UL)
        self.add(min1)
        self.add(*dots,*m_simplices)
        M = Matrix([
            ['S_{1,1}', 'S_{1,2}' ],
            ['S_{2,1}', 'S_{2,2}']
        ]).next_to(min1,DOWN,buff2,LEFT)
        self.add(M)
        M_entr = M.get_entries()
        r1 = SurroundingRectangle(M_entr[0])
        M_tex1 = [
            setColors(MathTex(r'M_{',str(i+1),',',str(j+1), r'}= -\int_',gen_omega_tex(),lang,r'\nabla',gen_fb_tex(),r'_',str(i+1),r'(x) \vert \nabla',gen_fb_tex(),'_',str(j+1),'(x)',rang, '\, dx')).next_to(M,DOWN,buff2,LEFT)
            for j in range(2) for i in range(2)
        ]
        M_tex2 = [
            setColors(MathTex(''.join([r'S_{',str(i+1),',',str(j+1), r'}= -\sum_{T \in \{A,B\}} {||e_',str(i+1),r'|| \over 2A_T} {||e_',str(j+1),r'|| \over 2A_T} \cos{\varTheta}']))).next_to(M,DOWN,buff2,LEFT)
            for j in range(2) for i in range(2)
        ]
        M_tex3 = [
            setColors(MathTex(''.join([r'S_{',str(i+1),',',str(j+1), r'}= - {1 \over 2} ( \cot{\alpha_{',str(i+1),str(j+1),r'}} + \cot{\beta_{',str(i+1),str(j+1),r'}))']))).next_to(M,DOWN,buff2,LEFT)
                if (i != j) else
            setColors(MathTex(''.join(
                [r'S_{', str(i + 1), ',', str(j + 1), r'}= {1 \over 2} \sum_{j \in N(i)} \cot{\alpha_{',str(i+1),str(j+1),r'}} + \cot{\beta_{',str(i+1),str(j+1),r'}']))).next_to(M, DOWN, buff2, LEFT)
            for j in range(2) for i in range(2)
        ]
        MathTex.set_default(font_size=0.15*DEFAULT_FONT_SIZE)
        M2 = Matrix([[M_tex3[0].tex_string,M_tex3[2].tex_string],[M_tex3[1].tex_string,M_tex3[3].tex_string]]).scale(3).next_to(min1,DOWN,buff2,LEFT)
        MathTex.set_default(font_size=DEFAULT_FONT_SIZE)
        m = M_tex1[0]
        self.play(Create(r1),Create(m),Create(fin_basis[0])); self.ns()
        nab_n = 10;
        Nab11 = [VGroup(
            *[ Arrow(d+np.cos(t*TAU/nab_n)*RIGHT+np.sin(t*TAU/nab_n)*UP,d).set_color_by_gradient(BLUE,BLACK) for t in range(nab_n)]
        ) for d in [dots[5].get_center(),dots[6].get_center(),dots[9].get_center(),dots[10].get_center()]]
        self.play(Create(Nab11[0])); self.ns()
        self.play(Transform(r1,SurroundingRectangle(M_entr[2])),Transform(m,M_tex1[1]),Create(Nab11[2]),Create(fin_basis[1])); self.ns()

        min2 = setColors(MathTex(r'|| \nabla',gen_fb_tex(),r'_i ||= 1/h = {||e_i|| \over 2A}')).next_to(m,DOWN,buff2,LEFT)
        self.play(FadeIn(min2)); self.ns();
        self.play(Transform(m,M_tex2[1])); self.ns()
        self.play(Transform(m,M_tex3[1]),FadeOut(min2)); self.ns()
        self.play(FadeOut(m),Transform(M,M2),FadeOut(r1),FadeOut(fin_basis[0]),FadeOut(fin_basis[1]),FadeOut(Nab11[0]),FadeOut(Nab11[2])); self.ns()
        self.play(FadeOut(M),*[FadeOut(i) for i in dots+m_simplices])

class p03_3(Slide):
    def ns(self):
        self.next_slide();
        # self.wait(0.1)
    def construct(self):
        min1 =  setColors( MathTex(r'\text{solve } ',gen_S(),gen_a(),r' = ',gen_b())).to_corner(UL)
        self.add(min1)
        im_a = ImageMobject(r'pics/geodesic_form_h.png')
        self.add(im_a)
        shift = 2
        a1 = Rectangle(color=BLACK,height=shift*2,width=shift*2,z_index=1,fill_opacity=1).shift(UP*shift+LEFT*shift)
        a2 = Rectangle(color=BLACK,height=shift*2,width=shift*2,z_index=1,fill_opacity=1).shift(UP*shift+RIGHT*shift)
        a3 = Rectangle(color=BLACK,height=shift*2,width=shift*2,z_index=1,fill_opacity=1).shift(DOWN*shift+LEFT*shift)
        a4 = Rectangle(color=BLACK,height=shift*2,width=shift*2,z_index=1,fill_opacity=1).shift(DOWN*shift+RIGHT*shift)
        self.add(a1,a2,a3,a4)
        self.wait(0.1); self.next_slide()
        self.play(FadeOut(a1)); self.next_slide()
        self.play(FadeOut(a2)); self.next_slide()
        self.play(FadeOut(a3)); self.next_slide()
        self.play(FadeOut(a4)); self.next_slide()


