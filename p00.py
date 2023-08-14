import copy
import math

from manim import *
# or: from manimlib import *
from manim_slides.slide import Slide
from dataclasses import dataclass


def x_tex():
    return r'\overline{x}'


def y_tex():
    return r'\overline{y}'


def f_tex():
    return r'{f}'


def pi_tex():
    return r'\pi'
def K_tex():
    return r'{K}'


def setColors(tex: MathTex):
    tex.set_color_by_tex(x_tex(), RED_C)
    tex.set_color_by_tex(y_tex(), GREEN_C)
    tex.set_color_by_tex(f_tex(), BLUE_C)
    tex.set_color_by_tex(pi_tex(), YELLOW)
    tex.set_color_by_tex(K_tex(), PURPLE)
    return tex


def camera():
    v = VGroup()
    v.add(Line(np.array([-1, -0.8, 0]), np.array([0.5, -0.8, 0])))
    v.add(Line(np.array([-1, 0.8, 0]), np.array([0.5, 0.8, 0])))
    v.add(Line(np.array([-1, 0.8, 0]), np.array([-1, -0.8, 0])))
    v.add(Line(np.array([0.5, 0.8, 0]), np.array([0.5, -0.8, 0])))
    v.add(Line(np.array([0, 0.0, 0]), np.array([1, 0.8, 0])))
    v.add(Line(np.array([0, 0.0, 0]), np.array([1, -0.8, 0])))
    v.add(Line(np.array([1, 0.8, 0]), np.array([1, -0.8, 0])))
    return v


def light():
    return Circle(1, color=YELLOW, fill_color=YELLOW, fill_opacity=1)


def newL(s, t, color=WHITE):
    return Line(np.array([s[0], s[1], 0]), np.array(t[0], t[1]), color=color)


# Scene of path, that we can expand on and transform
@dataclass
class Scene:
    c = camera()
    l = light().next_to(c, RIGHT, buff=12)
    img_line = Line(np.array([0, -1, 0]), np.array([0, 1, 0])).next_to(c, RIGHT, buff=1)
    geom1 = Line(np.array([-1, 0, 0]), np.array([1, 0, 0])).next_to(img_line, UP + RIGHT * 2, buff=1)
    geom2 = Line(np.array([-1, 0, 0]), np.array([1, 0, 0])).next_to(geom1, DOWN * 2 + RIGHT, buff=2)
    lpath = VGroup(
        Line(c.get_right(), geom1.get_center(), color=RED),
        Line(geom1.get_center(), geom2.get_center(), color=RED),
        Line(geom2.get_center(), l.get_center(), color=RED),
    ).set_z_index(-1)
    font_size = DEFAULT_FONT_SIZE * 1.3
    tpath = VGroup(
        setColors(MathTex(x_tex(), '_0', font_size=font_size)).next_to(l, LEFT),
        setColors(MathTex(x_tex(), '_1', font_size=font_size)).next_to(geom2, UP,
                                                                       buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 1.5),
        setColors(MathTex(x_tex(), '_2', font_size=font_size)).next_to(geom1, DOWN),
        setColors(MathTex(x_tex(), '_3', font_size=font_size)).next_to(c, DOWN),
        z_index=2
    )
    lpath2 = VGroup(
        Line(c.get_right(), geom2.get_center()+RIGHT*0.4, color=GREEN),
        Line(geom2.get_center()+RIGHT*0.4, l.get_center(), color=GREEN),
    ).set_z_index(-1)
    tpath2 = VGroup(
        setColors(MathTex(y_tex(), '_0', font_size=font_size)).next_to(l, LEFT+DOWN),
        setColors(MathTex(y_tex(), '_1', font_size=font_size)).next_to(geom2, DOWN),
        setColors(MathTex(y_tex(), '_2', font_size=font_size)).next_to(c, UP),
        z_index=2
    )
    lpathbd1 = VGroup(
        Line(c.get_right(), geom1.get_center(), color=RED,stroke_opacity=0.4),
        Line(geom1.get_center(), geom2.get_center(), color=RED,stroke_opacity=0.4),
        Line(geom2.get_center(), l.get_center(), color=RED,stroke_opacity=0.4),
    ).set_z_index(-1)
    tpathbd1 = VGroup(
        setColors(MathTex(x_tex(), '_0', font_size=font_size,opacity=1)).next_to(l, LEFT),
        setColors(MathTex(x_tex(), '_1', font_size=font_size,fill_opacity=0.4)).next_to(geom2, UP,
                                                                       buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 1.5),
        setColors(MathTex(x_tex(), '_2', font_size=font_size,fill_opacity=0.4)).next_to(geom1, DOWN),
        setColors(MathTex(x_tex(), '_3', font_size=font_size)).next_to(c, DOWN),
        z_index=2
    )
    lpathlp1 = VGroup(
        Line(c.get_right(), geom1.get_center(), color=RED,stroke_opacity=0.4),
        Line(geom1.get_center(), geom2.get_center(), color=RED,stroke_opacity=0.4),
        Line(geom2.get_center(), l.get_center(), color=RED,stroke_opacity=1.0),
    ).set_z_index(-1)
    tpathlp1 = VGroup(
        setColors(MathTex(x_tex(), '_0', font_size=font_size,fill_opacity=1)).next_to(l, LEFT),
        setColors(MathTex(x_tex(), '_1', font_size=font_size,fill_opacity=1.0)).next_to(geom2, UP,
                                                                                        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 1.5),
        setColors(MathTex(x_tex(), '_2', font_size=font_size,fill_opacity=0.4)).next_to(geom1, DOWN),
        setColors(MathTex(x_tex(), '_3', font_size=font_size,fill_opacity=0.4)).next_to(c, DOWN),
        z_index=2
    )
    lpathlp2 = VGroup(
        Line(c.get_right(), geom1.get_center()+LEFT*0.5, color=GREEN,stroke_opacity=1.0),
        DashedLine(geom1.get_center()+LEFT*0.5, geom2.get_center(), color=GREEN,stroke_opacity=1.0),
        Line(geom2.get_center(), l.get_center(), color=GREEN,stroke_opacity=1.0),
    ).set_z_index(-1)
    tpathlp2 = VGroup(
        setColors(MathTex(y_tex(), '_0', font_size=font_size)).next_to(l, LEFT+DOWN),
        setColors(MathTex(y_tex(), '_1', font_size=font_size)).next_to(geom2, DOWN),
        setColors(MathTex(y_tex(), '_2', font_size=font_size)).next_to(geom1, UP),
        setColors(MathTex(y_tex(), '_3', font_size=font_size)).next_to(c, UP),
        z_index=2
    )
    lpathcp2 = VGroup(
        DashedLine(c.get_right(), geom1.get_center()+LEFT*0.5, color=GREEN,stroke_opacity=1.0),
        Line(geom1.get_center()+LEFT*0.5, geom2.get_center(), color=GREEN,stroke_opacity=1.0),
        Line(geom2.get_center(), l.get_center(), color=GREEN,stroke_opacity=1.0),
    ).set_z_index(-1)
    tpathcp2 = VGroup(
        setColors(MathTex(y_tex(), '_0', font_size=font_size)).next_to(l, LEFT+DOWN),
        setColors(MathTex(y_tex(), '_1', font_size=font_size)).next_to(geom2, DOWN),
        setColors(MathTex(y_tex(), '_2', font_size=font_size)).next_to(geom1, UP),
        setColors(MathTex(y_tex(), '_3', font_size=font_size)).next_to(c, UP),
        z_index=2
    )
    base = RoundedRectangle(stroke_opacity=1,stroke_width=1, fill_opacity=0, height=9, width=18).move_to(RIGHT * 6 + RIGHT)
    Polygon.set_default(z_index=-1, stroke_color=YELLOW_B, fill_color=YELLOW_A, fill_opacity=0.4, stroke_opacity=0.8)
    ap1 = Polygon(c.get_right() + DOWN * 0.8, c.get_right() + UP * 0.35, geom1.get_left(), geom1.get_right())
    ap2 = Polygon(geom1.get_left(), geom1.get_right(), l.get_top(), l.get_bottom())
    ap3 = Polygon(c.get_right() + DOWN * 0.8, c.get_right() + UP * 0.8, geom2.get_right(), geom2.get_left())
    ap4 = Polygon(geom2.get_right(), geom2.get_left(), l.get_top(), l.get_bottom())
    ap5 = Polygon(c.get_right() + DOWN * 0.4, c.get_right() + UP * 0.4, l.get_top(), l.get_bottom())
    ap6 = Polygon(geom1.get_left(), geom1.get_right(), geom2.get_right(), geom2.get_left())
    measContrFactors = VGroup(
        setColors(MathTex("L_e(x_0,x_1)", font_size=font_size).next_to(lpath[2].get_right(), DOWN)),
        setColors(MathTex("f_s(x_0,x_1,x_2)", font_size=font_size).next_to(geom2.get_center(), DOWN)),
        setColors(MathTex("G(x_0,x_1)", font_size=font_size).move_to(lpath[2].get_center())),
        setColors(MathTex("f_s(x_1,x_2,x_3)", font_size=font_size).next_to(geom1.get_center(), UP)),
        setColors(MathTex("G(x_1,x_2)", font_size=font_size).move_to(lpath[1].get_center())),
        setColors(MathTex("G(x_2,x_3)", font_size=font_size).move_to(lpath[0].get_center())),
        setColors(MathTex("W_e(x_2,x_3)", font_size=font_size).next_to(c, UP)),
    )
    ap12Ar = VGroup(
        Line(c.get_right()+DOWN*0.8,c.get_right()+UP*0.35,color=RED),
        Line(geom1.get_left(), geom1.get_right(),color=RED),
        Line(geom2.get_left(), geom2.get_right(),color=RED),
        Arc(l.radius,start_angle=1/2*math.pi,angle=math.pi,arc_center=l.get_center(),color=RED),
        z_index=3
    )


    def withBase(self):
        return VGroup(self.l, self.c, self.img_line, self.geom1, self.geom2, self.base).copy()

    def withPath(self):
        b = self.withBase()
        b.add(self.lpath, self.tpath)
        return b.copy()
    def with2Path(self):
        b = self.withBase()
        b.add(self.lpath, self.tpath,self.lpath2,self.tpath2)
        return b.copy()

    def withMeasContrFunc(self):
        b = self.withBase()
        b.add(self.lpath, self.tpath, self.measContrFactors)
        return b.copy()

    def withImgContr(self):
        b = self.withBase()
        b.add(self.lpath, self.ap1, self.ap2, self.ap3, self.ap4, self.ap5, self.ap6)
        return b.copy()

    def withPrMeas(self):
        b = self.withBase()
        b.add(self.lpath, self.ap1, self.ap4,self.ap6,self.ap12Ar)
        return b.copy()
    def withBD1(self):
        b = self.withBase()
        b.add(self.lpathbd1, self.tpathbd1)
        return b.copy()
    def withBD2(self):
        b = self.withBase()
        b.add(self.lpathbd1,self.tpathbd1,self.lpath2,self.tpath2)
        return b.copy()
    def withLP1(self):
        b = self.withBase()
        b.add(self.lpathlp1, self.tpathlp1)
        return b.copy()
    def withLP2(self):
        b = self.withBase()
        b.add(self.lpathlp1, self.tpathlp1,self.lpathlp2, self.tpathlp2)
        return b.copy()
    def withCP1(self):
        b = self.withBase()
        b.add(self.lpathlp1, self.tpathlp1)
        return b.copy()
    def withCP2(self):
        b = self.withBase()
        b.add(self.lpathlp1, self.tpathlp1,self.lpathcp2, self.tpathcp2)
        return b.copy()


def transScene(g: VGroup):
    return g.scale(0.361).to_corner(UP + RIGHT)


wait_time = 1
write_time = 1.0
doub_buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 2


class p00_0(Slide):

    def intro(self):
        def tbwrittenOut():
            return ['\int_\Omega', f_tex(), '(',x_tex(),r') \, d \mu( ', x_tex(), r')']
        tmce = Tex(r'\underline{Monte Carlo Estimator}').scale(2).to_edge(UP)
        self.add(tmce)
        self.wait(); self.next_slide()
        tmc = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'{',f_tex(),r'(',x_tex(),r'^i)',r' \over ',r'p(',x_tex(),r'^i)} ',r'\right]'))
        self.play(FadeIn(tmc))
        self.wait(wait_time); self.next_slide()
        framebox1 = SurroundingRectangle(tmc[7:10], buff = .02,fill_opacity=0,stroke_color=YELLOW)
        self.play(Create(framebox1))
        self.wait(wait_time); self.next_slide()
        tmlt = Tex(r'\underline{Metropolis Light Transport}').scale(2).to_edge(UP)
        self.wait(wait_time); self.next_slide()
        self.play(Transform(tmce,tmlt))
        self.wait(wait_time); self.next_slide()
        tmc3 = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'{',f_tex(),'(',x_tex(),r'^i)',r' \over ',r'{1 \over',*tbwrittenOut(),r'} \cdot ',f_tex(),'(',x_tex(),r'^i)}',r'\right]'))
        framebox2 = SurroundingRectangle(tmc3[6:16], buff = .02,fill_opacity=0,stroke_color=YELLOW)
        self.play(TransformMatchingTex(tmc,tmc3),Transform(framebox1,framebox2))
        self.wait(wait_time); self.next_slide()
        tmc4 = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'',*tbwrittenOut(),r'\right]'))
        self.play(TransformMatchingTex(tmc3,tmc4),FadeOut(framebox2),FadeOut(framebox1))
        self.wait(wait_time); self.next_slide()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()


    def initialExample(self):
        states = []
        probs = []
        ptexts = []
        dil = 3.5
        vg = VGroup()
        for i in range(0, 4):
            pos = LEFT * (dil - dil * i) + LEFT * 2 + UP * 1
            t = MathTex("x_" + str(i)).shift(pos)
            states.append(t)
            vg.add(t)

            p = 0
            if i == 2:
                p = 1
            probs.append(p)
            pt = MathTex("p_0", "(", "x_" + str(i), ") =", str(p)).shift(pos + DOWN * 2)
            pt.set_color_by_tex("x_", t.color)
            ptexts.append(pt)
            if i < 3:
                vg.add(CurvedArrow(pos + DOWN * 0.2 + RIGHT * 0.2, pos + DOWN * 0.2 + LEFT * 0.2 + RIGHT * dil, angle=0.5))
                vg.add(CurvedArrow(pos + DOWN * 0.2 + LEFT * 0.2 + RIGHT * dil, pos + DOWN * 0.2 + RIGHT * 0.2,  angle=-0.5))
            else:
                vg.add(CurvedArrow(pos + UP * 0.2, pos + UP * 0.2 + LEFT * dil * 3, angle=1.2))
                vg.add(CurvedArrow(pos + UP * 0.2 + LEFT * dil * 3,pos + UP * 0.2,  angle=-1.2))
        actual_state = Dot(color=RED).next_to(states[2],DOWN)
        self.play(Create(vg))
        self.next_slide()
        self.play(states[2].animate.set_color(RED_D), Write(ptexts[2]),Create(actual_state))
        self.next_slide()  # Waits user to press continue to go to the next slide
        self.play(Write(ptexts[0]), Write(ptexts[1]), Write(ptexts[3]))
        self.next_slide()  # Waits user to press continue to go to the next slide

        def mtFrac(i, j):
            return MathTex("\\frac{" + str(i) + "}{" + str(j) + "}", font_size=DEFAULT_FONT_SIZE / 2)

        t0 = mtFrac(0, 3).shift(states[1].get_center() + UP * 1.0)
        t1 = mtFrac(1, 3).shift(states[1].get_center() + dil / 2 * RIGHT + UP * 0.05)
        arc = CurvedArrow(states[2].get_center() + UP * 0.2 + RIGHT * 0.2,
                               states[2].get_center() + UP * 0.2 + LEFT * 0.2, angle=4,color=RED)
        arc0 = CurvedArrow(states[2].get_center() + UP * 0.2 + LEFT * 0.2,
                                states[0].get_center() + UP * 0.2 + RIGHT * 0.2, angle=0.6, stroke_opacity=0.5,tip_style={"fill_opacity":0.5})
        t2 = mtFrac(1, 3).shift(states[2].get_center() + UP * 0.9)
        t3 = mtFrac(1, 3).shift(states[2].get_center() + dil / 2 * RIGHT + UP * 0.05)
        newpt = [t0, t1, t2, t3]
        self.play(Create(arc), Create(arc0), vg[5].animate.set_color(RED), vg[7].animate.set_color(RED))
        vg[7].z_index=1
        self.play(FadeIn(t1), FadeIn(t2), FadeIn(t3), FadeIn(t0))
        self.wait(1); self.next_slide()
        iter = [ptexts[0].submobjects[0], ptexts[1].submobjects[0], ptexts[2].submobjects[0], ptexts[3].submobjects[0]]
        newIt = iter.copy()

        def get_Mv(txt_idx):
            return newpt[txt_idx].animate.move_to(ptexts[txt_idx].submobjects[4].get_center())

        def anPIt(txt_idx, iteration):
            if iteration == -1:
                newIt[txt_idx] = MathTex("p_", "\\infty").move_to(iter[txt_idx].get_center() + LEFT * 0.05)
            else:
                newIt[txt_idx] = MathTex("p_" + str(iteration)).move_to(iter[txt_idx].get_center())
            return TransformMatchingShapes(iter[txt_idx], newIt[txt_idx])

        def getFo(i):
            return FadeOut(ptexts[i].submobjects[4])

        def anStateColTxt(i, color):
            return states[i].animate.set_color(color)


        self.play(get_Mv(0), get_Mv(1), get_Mv(2), get_Mv(3), getFo(0), getFo(1), getFo(2), getFo(3),
                  anStateColTxt(1, RED_B), anStateColTxt(2, RED_B), anStateColTxt(3, RED_B), FadeOut(arc),
                  FadeOut(arc0), actual_state.animate.next_to(states[1],DOWN),
                  vg[5].animate.set_color(WHITE), vg[7].animate.set_color(WHITE),
                  anPIt(0, 1), anPIt(1, 1), anPIt(2, 1), anPIt(3, 1))
        self.wait(1); self.next_slide();
        iter = newIt.copy()  # Waits user to press continue to go to the next slide
        arc = CurvedArrow(states[1].get_center() + UP * 0.2 + RIGHT * 0.2, states[1].get_center() + UP * 0.2 + LEFT * 0.2, angle=4,color=RED)
        vg[4].z_index=1
        self.play(
            Create(arc),vg[2].animate.set_color(RED), vg[4].animate.set_color(RED)
        )
        nt0 = mtFrac(2, 9).move_to(t0.get_center())
        nt1 = mtFrac(2, 9).move_to(t1.get_center())
        nt2 = mtFrac(3, 9).move_to(t2.get_center())
        nt3 = mtFrac(2, 9).move_to(t3.get_center())
        self.play(TransformMatchingShapes(t0, nt0), TransformMatchingShapes(t1, nt1), TransformMatchingShapes(t2, nt2),
                  TransformMatchingShapes(t3, nt3), actual_state.animate.next_to(states[0],DOWN),
                  anStateColTxt(0, RED_A), anStateColTxt(1, RED_A), anStateColTxt(2, RED_B), anStateColTxt(3, RED_A),
                  FadeOut(arc),vg[2].animate.set_color(WHITE), vg[4].animate.set_color(WHITE),
                  anPIt(0, 2), anPIt(1, 2), anPIt(2, 2), anPIt(3, 2))
        self.wait(1);
        self.next_slide();
        iter = newIt.copy()  # Waits user to press continue to go to the next slide

        nnt0 = mtFrac(2, 8).move_to(nt0.get_center())
        nnt1 = mtFrac(2, 8).move_to(nt1.get_center())
        nnt2 = mtFrac(2, 8).move_to(nt2.get_center())
        nnt3 = mtFrac(2, 8).move_to(nt3.get_center())
        self.play(TransformMatchingShapes(nt0, nnt0), TransformMatchingShapes(nt1, nnt1),
                  TransformMatchingShapes(nt2, nnt2), TransformMatchingShapes(nt3, nnt3),
                  actual_state.animate.next_to(states[1],DOWN),FadeOut(actual_state),
                  anStateColTxt(0, WHITE), anStateColTxt(1, WHITE), anStateColTxt(2, WHITE), anStateColTxt(3, WHITE),
                  anPIt(0, -1), anPIt(1, -1), anPIt(2, -1), anPIt(3, -1))
        self.wait(1);
        self.next_slide();
        iter = newIt  # Waits user to press continue to go to the next slide
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def formalise(self):
        self.clear()
        axes = Axes(
            x_range=[0, 10, 10],
            y_range=[0, 1, 10],
            x_length=5, y_length=3,
        ).to_corner(UP + LEFT)
        sst = Tex(r'State Space $\Omega$').next_to(axes, DOWN, aligned_edge=LEFT)
        self.play(Create(axes), Write(sst), run_time=write_time)
        self.wait(wait_time);
        self.next_slide()
        t = ValueTracker(3)
        dot = Dot(color=RED_C).move_to(axes.c2p(5, 0))
        set = Tex(r'elements ', r'$\overline{x}$', '$\, \in \Omega$').next_to(sst, DOWN, aligned_edge=LEFT)
        set.submobjects[1].color = RED_C
        self.play(Create(dot), Write(set), run_time=write_time)
        self.wait(wait_time);
        self.next_slide()

        tf = Tex(r'contribution Function ', '${f}$').next_to(set, DOWN, aligned_edge=LEFT)
        setColors(tf)
        tfm = MathTex(f_tex(), r'\,:\, \Omega \, \to \, \mathbb{R}').next_to(tf, DOWN, aligned_edge=LEFT)
        setColors(tfm)
        pf = axes.plot(lambda x: 0.6, color=BLUE_C)
        self.play(Write(tf), run_time=write_time)
        self.play(Write(tfm), Create(pf), run_time=write_time)
        self.wait(wait_time);
        self.next_slide()

        ttf = MathTex(r'\text{Transition Function:}', K_tex(),'(', y_tex(), ',', x_tex(), r')').to_corner(UP + RIGHT)
        setColors(ttf)
        ttfs = MathTex(r'\text{s.t.\,} \int_\Omega ',K_tex(),'(', y_tex(), ',', x_tex(), r') \, d \mu(', y_tex(), ') = 1')
        setColors(ttfs)
        ttfs.next_to(ttf, DOWN, aligned_edge=LEFT)
        ptf = axes.plot_line_graph(
            x_values=[0, 3.4999, 3.5, 6.5, 6.51, 10],
            y_values=[0, 0, 1 / 3.0, 1 / 3.0, 0, 0],
            line_color=PURPLE,
            vertex_dot_radius=0.0
        )
        areax1 = axes.c2p(3.5,0); areax2 = axes.c2p(6.5,1/3.0); areadiff = areax2-areax1
        area = Rectangle(height=areadiff[1],width=areadiff[0], fill_color=PURPLE_A, fill_opacity=0.2,stroke_opacity=0).move_to(areax1+areadiff/2)

        doty = Dot(color=GREEN).move_to(axes.c2p(t.get_value(), 0))
        doty.add_updater(lambda x: x.move_to(axes.c2p(t.get_value(), 0)))
        self.play(Write(ttf), Create(ptf), Create(doty),run_time=write_time)
        self.play(t.animate.set_value(1))
        self.play(t.animate.set_value(8))
        self.play(t.animate.set_value(6))
        self.wait();self.next_slide()
        self.play(Write(ttfs), Create(area),run_time=write_time)
        self.wait(wait_time);
        self.next_slide()

        tsd1 = Tex('Apply K until convergance').next_to(ttfs, DOWN, aligned_edge=LEFT,
                                                        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 3)
        tsd2 = Tex(r'to stationary distribution ', r'$\pi$').next_to(tsd1, DOWN, aligned_edge=LEFT)
        tsdm = MathTex('p_i(', x_tex(), r') = p_{i-1}(', x_tex(), ')').next_to(tsd2, DOWN, aligned_edge=LEFT,
                                                                               buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 2)
        tsd3 = Tex(r's.t. ', r'$\pi$', ' is proportional to ', '${f}$').next_to(tsdm, DOWN, aligned_edge=LEFT,
                                                                                buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 2)
        setColors(tsdm);
        setColors(tsd2);
        setColors(tsd3)
        self.play(Write(tsd1), run_time=write_time)
        self.play(Write(tsd2), run_time=write_time)
        self.play(Write(tsdm), run_time=write_time)
        self.wait(wait_time);
        self.next_slide()
        psd = axes.plot(lambda x: 0.1, color=YELLOW)
        self.play(Write(tsd3), run_time=write_time)
        self.play(Transform(ptf, psd),FadeOut(area))
        self.wait(wait_time);
        self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()

    def metropolis_sampling(self):
        tstate_space = Tex(r'$\Omega$', ": space of all paths").to_corner(UP + LEFT)
        tx = setColors(MathTex(r'\text{Path: }', x_tex(), r'\in \Omega').next_to(tstate_space, DOWN, aligned_edge=LEFT))

        def transScene(g: VGroup):
            return g.scale(0.4).to_corner(UP + RIGHT)

        bs = transScene(Scene().withBase())
        self.play(Create(bs), Write(tstate_space))
        self.wait(wait_time);
        self.next_slide()

        ps = transScene(Scene().withPath())
        self.play(TransformMatchingShapes(bs, ps), Write(tx))
        self.wait(wait_time);
        self.next_slide()

        tf = setColors(Tex('$' + f_tex() + '$', ': img. contribution func.')).next_to(tx, DOWN, buff=doub_buff,
                                                                                      aligned_edge=LEFT)
        tp1 = setColors( MathTex(r'\text{s.t. }', r'\int_\Omega', f_tex(), '_i(', x_tex(), r')\,', 'd \mu (', x_tex(), ')')).next_to(tf, DOWN, buff=doub_buff, aligned_edge=LEFT)
        self.play(Write(tf)); self.wait(wait_time); self.next_slide()
        tp2 = setColors(Tex(r'reflects the power flowing to image plane')).next_to(tp1, DOWN, buff=doub_buff,
                                                                                   aligned_edge=LEFT)
        self.play(Write(tp1), Write(tp2));
        bs = transScene(Scene().withImgContr())
        self.play(FadeTransform(ps, bs))
        self.wait(wait_time); self.next_slide()

        tfs = MathTex()
        tfs.submobjects = copy.deepcopy(tp1.submobjects[2:6])
        self.play(tfs.animate.next_to(tp2, DOWN, doub_buff, aligned_edge=LEFT))
        tfst = Tex("  : measurement contribution function").next_to(tfs)
        tmcf = MathTex(r'f_i(x_0, \ldots ,x_k) = L_e(x_0 \rightarrow x_1) \cdot \left( \prod_{j = 1}^{k-1} G\left(x_{j-1} \leftrightarrow x_j \right) \cdot f_s \left( x_{j-1} \rightarrow x_j \rightarrow x_{j+1} \right) \right) \cdot G\left(x_{k-1} \leftrightarrow x_{k} \right) \cdot W_e^j \left(x_{k-1} \rightarrow x_{k}\right)')
        tmcf.font_size = DEFAULT_FONT_SIZE / 2
        tmcf.next_to(tfs, DOWN, aligned_edge=LEFT)
        ps = transScene(Scene().withMeasContrFunc())
        self.play(FadeIn(tfst),Write(tmcf), FadeTransform(bs, ps))
        self.wait(wait_time); self.next_slide()

        tpm = MathTex()
        tpm.submobjects = copy.deepcopy(tp1.submobjects[6:])
        self.play(FadeOut(tmcf), tpm.animate.next_to(tfs, DOWN, aligned_edge=LEFT))
        tpmt = Tex(": product measure").next_to(tpm)
        tpmf = setColors(MathTex(r'd \mu(',x_tex(),r'_0, \ldots, ',x_tex(),r'_k) = dA(',x_tex(),r'_0) \cdot \ldots \cdot dA(',x_tex(),r'_k)').next_to(tpmt,RIGHT))
        bs = transScene(Scene().withPrMeas())
        self.play(FadeIn(tpmt),FadeTransform(ps,bs),FadeIn(tpmf))
        self.wait(wait_time); self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()

    def transition_function(self):
        ttf = setColors(MathTex(r'\text{Transition Function: }', r'K(', y_tex(), ',', x_tex(), r')').to_corner(UP + LEFT))
        scene1 = transScene(Scene().withPath())
        scene2 = transScene(Scene().with2Path())
        self.play(FadeIn(scene1),Write(ttf))
        self.wait(wait_time); self.next_slide()
        self.play(FadeTransform(scene1,scene2))
        self.wait(wait_time); self.next_slide()

        ttf2 = setColors(MathTex(r'\text{s.t. }', pi_tex(), r'\text{ is proportional to }',f_tex())).next_to(ttf,DOWN,aligned_edge=LEFT)
        self.play(Write(ttf2)); self.wait(wait_time); self.next_slide()

        ttf3 = setColors(Tex("but ", "$"+f_tex()+"$"," can only be sampled")).next_to(ttf2,DOWN,aligned_edge=LEFT)
        self.play(Write(ttf3)); self.wait(wait_time); self.next_slide()

        def tentativeFunc():
            return [r'T(',y_tex(),',',x_tex(),')']
        def tentativeFuncr():
            return [r'T(',x_tex(),',',y_tex(),')']
        tk = MathTex()
        tk.submobjects = copy.deepcopy(ttf.submobjects[1:])
        self.play(tk.animate.move_to(ORIGIN+DOWN*2))
        tentFunc = tentativeFunc()
        tk2 =setColors(MathTex(*tentFunc,r'\cdot a(',y_tex(),',',x_tex(),r')')).move_to(tk)
        tk2b1 = Brace(tk2[:5],UP)
        tk2b1l = tk2b1.get_text("tentative transition function")
        tk2b2 = Brace(tk2[6:],DOWN)
        tk2b2l = tk2b2.get_text("acceptance probability")
        tk2bvg = VGroup(tk2b1,tk2b1l,tk2b2,tk2b2l)
        self.play(TransformMatchingTex(tk,tk2),FadeIn(tk2bvg))

        self.wait(wait_time); self.next_slide()
        tk3 =setColors(MathTex(
            *tentativeFunc(),r'\cdot \text{min}\left( 1, { { ',
            f_tex(),'(',y_tex(),r')\cdot',*tentativeFuncr(),r' \over',
            f_tex(),'(',x_tex(),r')\cdot',*tentativeFunc(),r'} }\right)',)).move_to(tk)
        tk3b1 = Brace(tk3[:5],UP)
        tk3b1l = tk3b1.get_text("tentative transition function")
        tk3b2 = Brace(tk3[5:],DOWN)
        tk3b2l = tk3b2.get_text("acceptance probability")
        tk3bvg = VGroup(tk3b1,tk3b1l,tk3b2,tk3b2l)
        self.play(TransformMatchingTex(tk2,tk3),TransformMatchingShapes(tk2bvg,tk3bvg))

        self.wait(wait_time); self.next_slide()
        tk4 = setColors(MathTex(r'K(', y_tex(), ',', x_tex(), r') =')).next_to(ttf3,DOWN,doub_buff,LEFT)
        tk2.next_to(tk4,RIGHT)
        self.play(TransformMatchingShapes(tk3,tk2),FadeOut(tk3bvg),FadeIn(tk4),FadeTransform(scene2,scene1))

        tttf = MathTex()
        tttf .submobjects = copy.deepcopy(tk2[0:5])
        tttf1 = copy.deepcopy(tttf)
        tbm2 = Tex("- bidirectional mutations").next_to(tk4,DOWN,doub_buff,LEFT)
        self.play(Transform(tttf1, tbm2))

        self.wait(wait_time); self.next_slide()
        scene2 = transScene(Scene().withBD1())
        self.play(FadeTransform(scene1,scene2))

        self.wait(wait_time); self.next_slide()
        scene1 = transScene(Scene().withBD2())
        self.play(FadeTransform(scene2,scene1))

        self.wait(wait_time); self.next_slide()
        scene2 = transScene(Scene().withPath())
        tttf2 = copy.deepcopy(tttf)
        tlp1 = Tex("- lens pertubation").next_to(tbm2,DOWN,aligned_edge=LEFT)
        self.play(Transform(tttf2, tlp1),FadeTransform(scene1,scene2))

        self.wait(wait_time); self.next_slide()
        scene1 = transScene(Scene().withLP1())
        self.play(FadeTransform(scene2,scene1))

        self.wait(wait_time); self.next_slide()
        scene2 = transScene(Scene().withLP2())
        self.play(FadeTransform(scene1,scene2))

        self.wait(wait_time); self.next_slide()
        scene1 = transScene(Scene().withPath())
        tttf3 = copy.deepcopy(tttf)
        tcp1 = Tex("- caustic pertubation").next_to(tlp1,DOWN,aligned_edge=LEFT)
        self.play(Transform(tttf3, tcp1),FadeTransform(scene2,scene1))

        self.wait(wait_time); self.next_slide()
        scene2 = transScene(Scene().withCP1())
        self.play(FadeTransform(scene1,scene2))

        self.wait(wait_time); self.next_slide()
        scene1 = transScene(Scene().withCP2())
        self.play(FadeTransform(scene2,scene1))

        self.wait(wait_time); self.next_slide()
        scene2 = transScene(Scene().withPath())
        tttf4 = copy.deepcopy(tttf)
        tmcp = Tex("- multi-chain pertubations").next_to(tcp1,DOWN,aligned_edge=LEFT)
        self.play(Transform(tttf4, tmcp),FadeTransform(scene1,scene2))

        self.wait(wait_time); self.next_slide()
        self.play(Transform(tttf1,tttf),Transform(tttf2,tttf),Transform(tttf3,tttf),Transform(tttf4,tttf))

        code = '''x <- initial_path()
image <- { array of zeros }
for i <- 1 to N
    y <- mutate(x)
    a = accept_prob(y,x)
    if random() < a
        then x <- y
    record_sample(image,x)'''
        rendered_code = Code(code=code, tab_width=4, font="Monospace", language="Python").to_corner(DOWN + RIGHT)

        self.wait(wait_time); self.next_slide()
        def tbwrittenOut():
            return ['\int_\Omega', f_tex(), '(',x_tex(),r') \, d \mu( ', x_tex(), r')']

        tmc1 = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'{f(',x_tex(),r'^i)',r' \over ',r'p(',x_tex(),r'^i)} ',r'\right]')).next_to(tk4,DOWN,doub_buff*4,LEFT).set_y(rendered_code.get_y())
        tmc2 = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'{f(',x_tex(),r'^i)',r' \over ',r'{1 \over b} \cdot f(',x_tex(),r'^i)}',r'\right]')).next_to(tk4,DOWN,doub_buff*4,LEFT).set_y(rendered_code.get_y())
        tmc3 = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'{f(',x_tex(),r'^i)',r' \over ',r'{1 \over',*tbwrittenOut(),r'} \cdot f(',x_tex(),r'^i)}',r'\right]')).next_to(tk4,DOWN,doub_buff*4,LEFT).set_y(rendered_code.get_y())
        tmc4 = setColors(MathTex(r'\mathbb{E} \left[{ 1 \over N }  \sum_{i=1}^N',r'',*tbwrittenOut(),r'\right]')).next_to(tk4,DOWN,doub_buff*2,LEFT).set_y(rendered_code.get_y())
        self.play(Write(tmc1))
        self.wait(wait_time); self.next_slide()
        self.play(TransformMatchingTex(tmc1,tmc2))
        self.wait(wait_time); self.next_slide()
        self.play(TransformMatchingTex(tmc2,tmc3))
        self.wait(wait_time); self.next_slide()
        self.play(TransformMatchingTex(tmc3,tmc4))
        self.wait(wait_time); self.next_slide()

        self.play(Write(rendered_code))
        self.wait(wait_time); self.next_slide()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()


    def construct(self):
        self.intro()
        self.initialExample()
        self.formalise()
        self.metropolis_sampling()
        self.transition_function()
