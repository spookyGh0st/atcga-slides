import copy
import math

import numpy as np
from manim import *
# or: from manimlib import *
from manim_slides.slide import Slide
from manim_slides.slide import ThreeDSlide
from dataclasses import dataclass


def x_tex():
    return r'\overline{x}'


def y_tex():
    return r'\overline{y}'


def f_tex():
    return r'{f}'


def pi_tex():
    return r'\pi'


def setColors(tex: MathTex):
    tex.set_color_by_tex(x_tex(), RED_C)
    tex.set_color_by_tex(y_tex(), GREEN_C)
    tex.set_color_by_tex(f_tex(), BLUE_C)
    tex.set_color_by_tex(pi_tex(), GREEN_B)
    return tex


@dataclass
class Vertex:
    p: np.ndarray
    dpdu: np.ndarray
    dpdv: np.ndarray
    n: np.ndarray
    dndu: np.ndarray
    dndv: np.ndarray
    eta: float = 1.0
    A: np.ndarray = None
    B: np.ndarray = None
    C: np.ndarray = None

@dataclass
class Vertex2d:
    p: np.ndarray
    dpdu: np.ndarray
    n: np.ndarray
    dndu: np.ndarray
    eta: float = 1.0
    A: float = None
    B: float = None
    C: float = None
    
@dataclass
class Ray2d:
    origin: np.ndarray
    direction: np.ndarray

    def __init__(self, vertex1: Vertex2d, vertex2: Vertex2d):
        self.origin = vertex1.p
        self.direction = vertex2.p - vertex1.p
        self.direction /= np.linalg.norm(self.direction)
        
    def intersect_circle(self, center: np.ndarray, radius: float) -> np.ndarray:
        oc = self.origin - center
        a = np.dot(self.direction, self.direction)
        b = 2.0 * np.dot(oc, self.direction)
        c = np.dot(oc, oc) - radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None  # No intersection

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        if t1 >= 0:
            return self.origin + t2 * self.direction
        elif t2 >= 0:
            return self.origin + t1 * self.direction
        else:
            return None  # Intersection behind the ray
    
    def intersect_line(self, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
        d_dot_n = np.dot(self.direction, normal)
        if d_dot_n == 0:
            return None  # Ray is parallel to the line

        t = np.dot(point - self.origin, normal) / d_dot_n
        if t >= 0:
            return self.origin + t * self.direction
        else:
            return None  # Intersection behind the ray

def reflect_on_circle(ray: Ray2d, center: np.ndarray, radius: float) -> Ray2d:
    intersection_point = ray.intersect_circle(center, radius)
    if intersection_point is None:
        return None  # No reflection if no intersection

    incident_dir = ray.direction
    normal = (intersection_point - center) / radius
    reflected_dir = incident_dir - 2 * np.dot(incident_dir, normal) * normal

    return Ray2d(Vertex2d(p=intersection_point, dpdu=None, n=None, dndu=None),
                 Vertex2d(p=intersection_point + reflected_dir, dpdu=None, n=None, dndu=None))



def computeHalfwayVec(vp: Vertex, vc: Vertex, vn: Vertex):
    wi = vp.p - vc.p
    wo = vn.p - vc.p
    ili = 1.0 / np.linalg.norm(wi)
    ilo = 1.0 / np.linalg.norm(wo)
    wi *= ili
    wo *= ilo
    H = wi + vc.eta * wo
    ilh = 1.0 / np.linalg.norm(H)
    H *= ilh
    return H


def computeDerivatives2d(vp: Vertex2d, vc: Vertex2d, vn: Vertex2d):
    # Compute relevant directions and a few useful projections
    wi = vp.p - vc.p
    wo = vn.p - vc.p
    ili = 1.0 / np.linalg.norm(wi)
    ilo = 1.0 / np.linalg.norm(wo)
    wi *= ili
    wo *= ilo
    H = wi + vc.eta * wo
    ilh = 1.0 / np.linalg.norm(H)
    H *= ilh
    dot_H_n = np.dot(vc.n, H)
    dot_H_dndu = np.dot(vc.dndu, H)
    dot_u_n = np.dot(vc.dpdu, vc.n)

    # Local shading tangent frame
    s = vc.dpdu - dot_u_n * vc.n
    ilo *= vc.eta * ilh
    ili *= ilh

    # Derivatives of C with respect to x_{i-1}
    dH_du = (vp.dpdu - wi * np.dot(wi, vp.dpdu)) * ili
    dH_du -= H * np.dot(dH_du, H)
    vc.A = np.dot(dH_du, s)

    # Derivatives of C with respect to x_i
    dH_du = -vc.dpdu * (ili + ilo) + wi * (np.dot(wi, vc.dpdu) * ili) + wo * (np.dot(wo, vc.dpdu) * ilo)
    dH_du -= H * np.dot(dH_du, H)
    vc.B = np.dot(dH_du, s) - np.dot(vc.dpdu, vc.dndu) * dot_H_n - dot_u_n * dot_H_dndu

    # Derivatives of C with respect to x_{i+1}
    dH_du = (vn.dpdu - wo * np.dot(wo, vn.dpdu)) * ilo
    dH_du -= H * np.dot(dH_du, H)
    vc.C = np.dot(dH_du, s)

def computeDerivatives(vp: Vertex, vc: Vertex, vn: Vertex):
    # Compute relevant directions and a few useful projections
    wi = vp.p - vc.p
    wo = vn.p - vc.p
    ili = 1.0 / np.linalg.norm(wi)
    ilo = 1.0 / np.linalg.norm(wo)
    wi *= ili
    wo *= ilo
    H = wi + vc.eta * wo
    ilh = 1.0 / np.linalg.norm(H)
    H *= ilh
    dot_H_n = np.dot(vc.n, H)
    dot_H_dndu = np.dot(vc.dndu, H)
    dot_H_dndv = np.dot(vc.dndv, H)
    dot_u_n = np.dot(vc.dpdu, vc.n)
    dot_v_n = np.dot(vc.dpdv, vc.n)

    # Local shading tangent frame
    s = vc.dpdu - dot_u_n * vc.n
    t = vc.dpdv - dot_v_n * vc.n
    ilo *= vc.eta * ilh
    ili *= ilh

    # Derivatives of C with respect to x_{i-1}
    dH_du = (vp.dpdu - wi * np.dot(wi, vp.dpdu)) * ili
    dH_dv = (vp.dpdv - wi * np.dot(wi, vp.dpdv)) * ili
    dH_du -= H * np.dot(dH_du, H)
    dH_dv -= H * np.dot(dH_dv, H)
    vc.A = np.array([
        [np.dot(dH_du, s), np.dot(dH_dv, s)],
        [np.dot(dH_du, t), np.dot(dH_dv, t)]
    ])

    # Derivatives of C with respect to x_i
    dH_du = -vc.dpdu * (ili + ilo) + wi * (np.dot(wi, vc.dpdu) * ili) + wo * (np.dot(wo, vc.dpdu) * ilo)
    dH_dv = -vc.dpdv * (ili + ilo) + wi * (np.dot(wi, vc.dpdv) * ili) + wo * (np.dot(wo, vc.dpdv) * ilo)
    dH_du -= H * np.dot(dH_du, H)
    dH_dv -= H * np.dot(dH_dv, H)
    vc.B = np.array([
        [np.dot(dH_du, s) - np.dot(vc.dpdu, vc.dndu) * dot_H_n - dot_u_n * dot_H_dndu,
         np.dot(dH_dv, s) - np.dot(vc.dpdu, vc.dndv) * dot_H_n - dot_u_n * dot_H_dndv],
        [np.dot(dH_du, t) - np.dot(vc.dpdv, vc.dndu) * dot_H_n - dot_v_n * dot_H_dndu,
         np.dot(dH_dv, t) - np.dot(vc.dpdv, vc.dndv) * dot_H_n - dot_v_n * dot_H_dndv]
    ])

    # Derivatives of C with respect to x_{i+1}
    dH_du = (vn.dpdu - wo * np.dot(wo, vn.dpdu)) * ilo
    dH_dv = (vn.dpdv - wo * np.dot(wo, vn.dpdv)) * ilo
    dH_du -= H * np.dot(dH_du, H)
    dH_dv -= H * np.dot(dH_dv, H)
    vc.C = np.array([
        [np.dot(dH_du, s), np.dot(dH_dv, s)],
        [np.dot(dH_du, t), np.dot(dH_dv, t)]
    ])


# Example usage:
x0 = Vertex(
    np.array([-1.0, 2.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
    np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
    1.0, None, None, None
)

x1 = Vertex(
    np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
    1.0, None, None, None
)

x2 = Vertex(
    np.array([2.0, 3.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
    np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
    1.0, None, None, None
)


computeDerivatives(x0, x1, x2)
A = x1.B
A = -np.linalg.inv(A)

# x3 is fixed, x1 is changing
B1 = x1.A
# x1 is fixed, x3 is changing
B3 = x1.C

# Derivative with respect to fixed and changing
ts1 = A * B1
ts3 = A * B3


vertices = [x0, x1, x2]
av0 = copy.deepcopy(x0)
av1 = copy.deepcopy(x1)
av2 = copy.deepcopy(x2)

tmp_l =0.5
lx0 = Vertex2d(
    np.array([-1. / math.sqrt(2), 1. / math.sqrt(2)]),np.array([-tmp_l,0]) ,
    np.array([0,-tmp_l]), np.array([0,0]) ,
    1.0,None, None, None
)
lx1 = Vertex2d(
    np.array([0,0]), np.array([tmp_l,0]) ,
    np.array([0,tmp_l]), np.array([tmp_l,0]) ,
    1.0,None, None, None
)
lx2 = Vertex2d(
    np.array([1. / math.sqrt(2), 1. / math.sqrt(2)]),np.array([0.0,tmp_l]) ,
    np.array([-tmp_l,0]), np.array([0,0]) ,
    1.5,None, None, None
)
lx3 = Vertex2d(
    np.array([1.5890238848747442,1.1785113019775793]),np.array([0.,tmp_l]) ,
    np.array([tmp_l,0]), np.array([0,0]) ,
    1.5,None, None, None
)
computeDerivatives2d(lx0,lx1,lx2)
computeDerivatives2d(lx1,lx2,lx3)
lDC = np.array([[lx1.A,lx1.B,lx1.C,0],[0,lx2.A,lx2.B,lx2.C]])
lB0 = lDC[:,0]
lB3 = lDC[:,3]
lB = lDC[:,[0,3]]

lA = lDC[:,[1,2]]
lInvA = -np.linalg.inv(lA)

# Derivative with respect to fixed and changing
lts1 = np.dot(lInvA, lB0)
lts3 = np.dot(lInvA, lB3)
lvertices = [lx0,lx1,lx2,lx3]
lax0 = copy.deepcopy(lx0)
lax1 = copy.deepcopy(lx1)
lax2 = copy.deepcopy(lx2)
lax3 = copy.deepcopy(lx3)
lavertices = [lax0,lax1,lax2,lax3]

# local projected altered x's
lpax0 = copy.deepcopy(lax0)
lpax1 = copy.deepcopy(lax1)
lpax2 = copy.deepcopy(lax2)
lpax3 = copy.deepcopy(lax3)
lpavertices = [lax0,lax1,lax2,lax3]

circle_center = np.array([0.0, -tmp_l])
circle_radius = tmp_l


# print(lx1)
# print(lx2)
print('========LDC==============')
print(lDC)
print('========LB0==============')
print(lB0)
print('========LA===============')
print(lA)
print('========LB3==============')
print(lB3)
print('========Lts1=============')
print(lts1)
print('========Lts3=============')
print(lts3)


@dataclass
class UVTracker:
    u: ValueTracker = ValueTracker(0)
    v: ValueTracker = ValueTracker(0)


class p01_0(Slide):
    scale = 1
    def construct(self):
        texheadline= Tex(r'\underline{Manifold Exploration}',font_size=1.5*DEFAULT_FONT_SIZE).to_corner(UP+LEFT)
        self.add(texheadline); self.wait();self.next_slide()
        coord = Axes(x_range=[-1.0, 2.0, 1],x_length=6*self.scale, y_range=[-0.5, 1.5, 1], y_length=4*self.scale,axis_config={"include_numbers": True,"z_index":-0})

        def up_p_of_v(p: Dot, v: Vertex2d):
            p.move_to(coord.c2p(v.p[0],v.p[1]))

        Dot.set_default(z_index = 5)
        p0 = Dot(color=RED); p1 = Dot(); p2 = Dot(); p3 = Dot()
        up_p_of_v(p0, lx0)
        up_p_of_v(p1, lx1)
        up_p_of_v(p2, lx2)
        up_p_of_v(p3, lx3)

        l01 = Line(p0.get_center(),p1.get_center(),z_index=-1,stroke_opacity=0.5)
        l12 = Line(p1.get_center(),p2.get_center(),z_index=-1,stroke_opacity=0.5)
        l23 = Line(p2.get_center(),p3.get_center(),z_index=-1,stroke_opacity=0.5)
        l01.add_updater(lambda l: l.put_start_and_end_on(p0.get_center(),p1.get_center()))
        l12.add_updater(lambda l: l.put_start_and_end_on(p1.get_center(),p2.get_center()))
        l23.add_updater(lambda l: l.put_start_and_end_on(p2.get_center(),p3.get_center()))



        tp0 = MathTex(r'D').next_to(p0,UP)
        tp1 = MathTex(r'S').next_to(p1,LEFT+DOWN)
        tp2 = MathTex(r'D').next_to(p2,LEFT+UP)
        tp3 = MathTex(r'D').next_to(p3,RIGHT)

        coordHR = coord.c2p(0.5,0) - coord.get_origin()
        coordHU = coord.c2p(0,0.5) - coord.get_origin()
        geom0 = Line(p0.get_center()-coordHR,p0.get_center()+coordHR,stroke_opacity=0.5,z_index=-5)
        geom1 = Line(p1.get_center()-coordHR,p1.get_center()+coordHR,stroke_opacity=0.5,z_index=-5)
        geom2 = Line(p2.get_center()-coordHU,p2.get_center()+coordHU,stroke_opacity=0.5,z_index=-5)
        geom3 = Line(p3.get_center()-coordHU,p3.get_center()+coordHU,stroke_opacity=0.5,z_index=-5)
        geometry= [geom0,geom1,geom2, geom2]
        vg = VGroup(coord,geom0,p0,tp0,l01,geom1,p1,tp1,l12,geom2,p2,tp2,l23,geom3,p3,tp3)
        self.play(Create(vg),run_time=4)
        self.wait(1); self.next_slide()
        self.play(p2.animate.shift(UP*1.0),p0.animate.shift(RIGHT*1.0),run_time = 2)
        self.play(p2.animate.shift(DOWN*2.0),p0.animate.shift(LEFT*2.0), run_time = 2)
        self.play(p2.animate.shift(UP*1.0),p0.animate.shift(RIGHT*1.0),run_time = 2)
        self.wait(1); self.next_slide()

        tp2n = MathTex(r'\boldsymbol{S}').next_to(p2,LEFT+UP)
        self.play(Transform(tp2,tp2n))
        self.wait(1); self.next_slide()


        def up_l23(l: Mobject):
            l.put_start_and_end_on(p2.get_center(), p3.get_center())
            y = coord.p2c(p2.get_center())[1]
            if (y <= lx2.p[1]-0.05 or y >= lx2.p[1]+0.05):
                l.set_color(RED_E)
            else:
                l.set_color(WHITE)

        self.remove(l23)
        l23 = Line(p2.get_center(),p3.get_center(),z_index=-1,stroke_opacity=0.5).add_updater(up_l23)
        self.add(l23)

        self.play(p2.animate.shift(UP*1.0),p0.animate.shift(RIGHT*1.0),run_time = 2)
        self.play(p2.animate.shift(DOWN*2.0),p0.animate.shift(LEFT*2.0), run_time = 2)
        self.play(p2.animate.shift(UP*1.0),p0.animate.shift(RIGHT*1.0),run_time = 2)
        self.wait(1); self.next_slide()

class p01_3(Slide):
    scale = 1

    def up_ax_on_tang(self, u0vt: ValueTracker ,u3vt:ValueTracker):
        u0 = u0vt.get_value(); u3 = u3vt.get_value()
        lax0.p = lx0.p + lx0.dpdu * u0
        lax0.n = lx0.n + lx0.dndu * u0

        lax1.p = lx1.p + lx1.dpdu * lts1[0]*u0 + lx1.dpdu*lts3[0]*u3
        lax1.n = lx1.n + lx1.dndu * lts1[0]*u0 + lx1.dndu*lts3[0]*u3

        lax2.p = lx2.p + lx2.dpdu * lts1[1]*u0 + lx2.dpdu*lts3[1]*u3
        lax2.n = lx2.n + lx2.dndu * lts1[1]*u0 + lx2.dndu*lts3[1]*u3

        lax3.p = lx3.p + lx3.dpdu * u3
        lax3.n = lx3.n + lx3.dndu * u3

        ray = Ray2d(lax2,lax1)
        lpax1.p = ray.intersect_circle(circle_center,circle_radius)
        refl_ray =  reflect_on_circle(ray,circle_center,circle_radius)
        ret = refl_ray.intersect_line(lx0.p,lx0.n)
        if type(ret) is np.ndarray:
            lpax0.p=ret
        else:
            lpax0.p = refl_ray.origin + refl_ray.direction*1


    def updateAltVerOnTang(self, u0, v0, u1, v1):
        av0.p = x0.p + x0.dpdu * u0.get_value() + x0.dpdv * v0.get_value()
        tuv1 = np.dot(ts1, np.array([u0.get_value(), v0.get_value()]))
        tuv2 = np.dot(ts3, np.array([u1.get_value(), v1.get_value()]))
        av1.p = x1.p + x1.dpdu * tuv1[0] + x1.dpdv * tuv2[1] + x1.dpdu * tuv2[0] + x1.dpdv * tuv2[1]*2
        av2.p = x2.p + x2.dpdu * u1.get_value() + x2.dpdv * v1.get_value()
        self.h = computeHalfwayVec(av0,av1,av2)

        # DEBUG
        # print("===========================================")
        # print(self.h)
        # print("uv1: "+ str(k0))
        # print("uv3: "+ str(k3))

        # print("===========================================")
        # print("p0: "+ str(av0.p))
        # print("p1: "+ str(av1.p))
        # print("p2: "+ str(av2.p))



    def cost_function(self):
        t = ValueTracker(math.pi/4)
        Arrow.set_default(z_index=1)
        Dot.set_default(z_index=2)
        coord = Axes(x_range=[-1.5, 1.5, 1],x_length=6*self.scale, y_range=[-1.5, 1.5, 1], y_length=6*self.scale,axis_config={"include_numbers": True}).to_edge(RIGHT)
        p0 = Dot(coord.c2p(-1./math.sqrt(2),1./math.sqrt(2)))
        p0.add_updater(lambda p: p.move_to(coord.c2p(-1./math.sqrt(2),1./math.sqrt(2))))
        p1 = Dot(coord.c2p(0,0))
        p1.add_updater(lambda p: p.move_to(coord.c2p(0,0)))
        tp1 = MathTex(x_tex(),'_i').next_to(p1,DOWN+LEFT)
        p2 = Dot(coord.c2p(math.sin(t.get_value()),math.cos(t.get_value())))
        p2.add_updater(lambda p: p.move_to(coord.c2p(math.sin(t.get_value()),math.cos(t.get_value()))))




        wi = Arrow(p1.get_center(),p0.get_center(),buff=0)
        def upWi(a:Arrow):
            a.put_start_and_end_on(p1.get_center(), p0.get_center())
        wi.add_updater(lambda a: upWi(a))
        wo = Arrow(p1.get_center(),p2.get_center(),buff=0)
        def upWo(a:Arrow):
            a.put_start_and_end_on(p1.get_center(), p2.get_center())
        wo.add_updater(lambda a: upWo(a))

        def upH(a:Arrow):
            if coord.p2c(wo.get_end())[1] > 0:
                refwi = 1.0
                refwo = 1.0
            else :
                refwi = 1.0
                refwo = 1.5
            hv = refwi*coord.p2c(wi.get_end()) + refwo*coord.p2c(wo.get_end())
            hv = hv * (1/ np.linalg.norm(hv))
            a.put_start_and_end_on(p1.get_center(), coord.c2p(hv[0],hv[1]))
        h = Arrow(buff=0,color=RED)
        upH(h)
        h.add_updater(lambda a: upH(a))
        b = Arrow(p1.get_center(),coord.c2p(1,0),color=BLUE,buff=0)
        b.add_updater(lambda a: a.put_start_and_end_on(p1.get_center(),coord.c2p(1,0)))

        def upC(d: DecimalNumber):
            f: float = np.dot(np.array([1, 0]), coord.p2c(h.get_end()))
            d.set_value(f)
            f = rate_functions.ease_out_cubic(np.abs(f))
            rgb_g = np.array([0,1,0])
            rgb_r = np.array([1,0,0])
            rgb_col = rgb_g*(1-f) + rgb_r*f
            wo.set_color(rgb_to_color(rgb_col))
            d.set_color(rgb_to_color(rgb_col))

        doub_buf = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*5
        cdescr = MathTex(r'\boldsymbol{c}_i(x) = ').to_corner(UP+LEFT)
        c = DecimalNumber(0,num_decimal_places=2,include_sign=True).next_to(cdescr,RIGHT)
        c.add_updater(lambda d: upC(d))
        htex = r'{{ {h} }} \left(\boldsymbol{x_i}, \overrightarrow{\boldsymbol{w}_i}, \overrightarrow{\boldsymbol{w}_o} \right)'
        twi = r'\overrightarrow{\boldsymbol{w}_i}'
        two = r'\overrightarrow{\boldsymbol{w}_o}'
        cdescr2 = MathTex(r'{{ {T} }}(\boldsymbol{x}_i)^T \cdot', htex, '=').next_to(cdescr,RIGHT)
        cdescr2.set_color_by_tex('{h}',h.color).set_color_by_tex('{T}',b.color)
        fresnel = [r'{\eta(',twi,')',twi,r'+ \eta(',two,')',two,'}']
        hdescr1 = MathTex(htex, r'= ',font_size=DEFAULT_FONT_SIZE)
        hdescr1.set_color_by_tex('{h}',h.color).next_to(cdescr,DOWN,doub_buf,aligned_edge=LEFT)
        hdescr2 = MathTex(r'{',*fresnel,r'\over',r'\left\Vert',*fresnel,r'\right\Vert }',font_size=DEFAULT_FONT_SIZE*(2.0/3)).next_to(hdescr1,RIGHT)
        bdescr1 = Tex('${T}$',': local basis',).set_color_by_tex('{T}',b.color).next_to(hdescr1,DOWN,doub_buf,aligned_edge=LEFT)
        bdescr2 = MathTex('{T}',r'^T =').set_color_by_tex('{T}',b.color).next_to(bdescr1,DOWN,doub_buf,aligned_edge=LEFT)
        bdescr3 = MathTex(r'\begin{bmatrix} 1 \\ 0 \end{bmatrix}^T = \begin{bmatrix} 1 & 0 \end{bmatrix}',font_size=DEFAULT_FONT_SIZE*(2.0/3)).next_to(bdescr2,RIGHT)


        self.add_updater(lambda t: upH(h))

        self.play(Create(coord),FadeIn(p0),FadeIn(p1),FadeIn(p2),FadeIn(tp1))
        self.wait(1);self.next_slide()
        self.play(Create(wi),Create(wo),FadeIn(cdescr),FadeIn(c))
        self.wait(1);self.next_slide()
        self.play(t.animate.set_value(math.pi*2/4 - 0.4908826782893113+math.pi/2),run_time = 2)
        self.wait(1);self.next_slide()
        self.play(t.animate.set_value(math.pi*(9/4)),run_time = 4)
        self.wait(1); self.next_slide()
        self.play(Write(cdescr2),c.animate(rate_func=rate_functions.ease_out_cubic).next_to(cdescr2,RIGHT),FadeIn(h),FadeIn(hdescr1),FadeIn(hdescr2),run_time=1)
        self.wait(1); self.next_slide()
        self.play(t.animate.set_value(math.pi*10/4 - 0.4908826782893113+math.pi/2),run_time = 2)
        self.wait(1); self.next_slide()
        self.play(FadeIn(b),Write(bdescr1),FadeIn(bdescr2),FadeIn(bdescr3))
        self.wait(1);self.next_slide()
        self.play(t.animate.set_value(math.pi*(17/4)),run_time = 4)
        self.wait(1); self.next_slide()
        self.play(FadeOut(hdescr1),FadeOut(hdescr2),FadeOut(bdescr1),FadeOut(bdescr2),FadeOut(bdescr3),
                  FadeOut(wi),FadeOut(wo),FadeOut(b),FadeOut(h),
                  FadeOut(c),FadeOut(cdescr),FadeOut(cdescr2), FadeOut(tp1),
                  run_time=1)
        # self.add(coord,p0,p1,p2)
        self.wait(1); self.next_slide()
        coord2 = Axes(x_range=[-1.0, 2.0, 1],x_length=6*self.scale, y_range=[-1.0, 2.0, 1], y_length=6*self.scale,axis_config={"include_numbers": True,"z_index":-0}).to_edge(RIGHT)
        coord2.z_index=-0
        self.play(coord.animate.scale(1.00).move_to(coord2.get_origin()))
        coord_copy = copy.deepcopy(coord)
        self.add(coord_copy); self.remove(coord)
        self.play(Transform(coord_copy,coord2))

        p3 = Dot(coord.c2p(coord.p2c(p2.get_center())[0]*3,-1/math.sqrt(2)))
        p3 = Dot(coord.c2p(1.5890238848747442,1.1785113019775793))
        tp0 = MathTex(r'\boldsymbol{x}_0').next_to(p0,UP)
        tp1 = MathTex(r'\boldsymbol{x}_1').next_to(p1,LEFT+DOWN)
        tp2 = MathTex(r'\boldsymbol{x}_2').next_to(p2,LEFT+UP)
        tp3 = MathTex(r'\boldsymbol{x}_3').next_to(p3,RIGHT)
        l01 = Line(p0.get_center(),p1.get_center(),z_index=-1,stroke_opacity=0.5)
        l12 = Line(p1.get_center(),p2.get_center(),z_index=-1,stroke_opacity=0.5)
        l23 = Line(p2.get_center(),p3.get_center(),z_index=-1,stroke_opacity=0.5)
        tp = VGroup(tp0,l01,tp1,l12,tp2,l23,p3,tp3)

        self.play(Create(tp))
        self.wait(1); self.next_slide()
        tc = r'\boldsymbol{c}'
        tx = r'\boldsymbol{x}'
        tC = MathTex(r'C = \begin{bmatrix} \phantom{c_i(x_i,x_i,x_i),x_i} \\ \phantom{c_i(x_i,x_i,x_i)} \end{bmatrix} = \boldsymbol{0}').to_corner(UP+LEFT)
        tC1 = MathTex(tc,'_1 (',tx,'_0,',tx,'_1,',tx,'_2)').move_to(tC).shift(RIGHT*0.0+UP*0.3)
        tC2 = MathTex(tc,'_2 (',tx,'_1,',tx,'_2,',tx,'_3)').move_to(tC).shift(RIGHT*0.0+DOWN*0.3)
        vgl12 = VGroup(copy.deepcopy(l01).set_color(RED).set_opacity(1),copy.deepcopy(l12).set_color(RED).set_opacity(1),z_index=2)
        vgl23 = VGroup(copy.deepcopy(l12).set_color(RED).set_opacity(1),copy.deepcopy(l23).set_color(RED).set_opacity(1),z_index=2)
        self.play(FadeIn(tC))
        self.next_slide(); self.add(vgl12); self.wait(1)
        self.play(Transform(vgl12,tC1))
        self.add(vgl23); self.wait()
        self.play(Transform(vgl23,tC2))
        self.wait(1); self.next_slide()
        tSS1 = Tex("Specular Manifold $S$").next_to(tC,DOWN,doub_buf,LEFT)
        tSS = MathTex(r'S = \left\{',x_tex(),'\, | \, C(',x_tex(),r') = \boldsymbol{0} \right\}').next_to(tSS1,DOWN,aligned_edge=LEFT)
        self.play(Write(tSS1),Write(tSS))

        self.wait(1); self.next_slide()
        self.play(FadeOut(tSS),FadeOut(tSS1))

        # self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()

    def create_c_der(self):
        coord = Axes(x_range=[-1.0, 2.0, 1],x_length=6*self.scale, y_range=[-1.0, 2.0, 1], y_length=6*self.scale,axis_config={"include_numbers": True,"z_index":-0}).to_edge(RIGHT)
        u0 = ValueTracker(0)
        u3 = ValueTracker(0)
        self.add_updater(lambda x, u0=u0,u3=u3: self.up_ax_on_tang(u0,u3))

        def up_p_of_v(p: Dot, v: Vertex2d):
            p.move_to(coord.c2p(v.p[0],v.p[1]))

        Dot.set_default(z_index = 5)
        p0 = Dot(); p1 = Dot(); p2 = Dot(); p3 = Dot()
        up_p_of_v(p0, lx0); p0.add_updater(lambda p: up_p_of_v(p, lax0))
        up_p_of_v(p1, lx1); p1.add_updater(lambda p: up_p_of_v(p, lax1))
        up_p_of_v(p2, lx2); p2.add_updater(lambda p: up_p_of_v(p, lax2))
        up_p_of_v(p3, lx3); p3.add_updater(lambda p: up_p_of_v(p, lax3))

        l01 = Line(p0.get_center(),p1.get_center(),z_index=-1,stroke_opacity=0.5)
        l12 = Line(p1.get_center(),p2.get_center(),z_index=-1,stroke_opacity=0.5)
        l23 = Line(p2.get_center(),p3.get_center(),z_index=-1,stroke_opacity=0.5)
        l01.add_updater(lambda l: l.put_start_and_end_on(p0.get_center(),p1.get_center()))
        l12.add_updater(lambda l: l.put_start_and_end_on(p1.get_center(),p2.get_center()))
        l23.add_updater(lambda l: l.put_start_and_end_on(p2.get_center(),p3.get_center()))

        tp0 = MathTex(r'\boldsymbol{x}_0').next_to(p0,UP)
        tp0.add_updater(lambda t: t.next_to(p0,UP))
        tp1 = MathTex(r'\boldsymbol{x}_1').next_to(p1,LEFT+DOWN)
        tp1.add_updater(lambda t: t.next_to(p1,LEFT+DOWN))
        tp2 = MathTex(r'\boldsymbol{x}_2').next_to(p2,LEFT+UP)
        tp2.add_updater(lambda t: t.next_to(p2,LEFT+UP))
        tp3 = MathTex(r'\boldsymbol{x}_3').next_to(p3,RIGHT)
        tp3.add_updater(lambda t: t.next_to(p3,RIGHT))

        coordHR = coord.c2p(0.5,0) - coord.get_origin()
        coordHU = coord.c2p(0,0.5) - coord.get_origin()
        geom0 = Line(p0.get_center()-coordHR,p0.get_center()+coordHR,stroke_opacity=0.5,z_index=-5)
        geom1 = Circle(2*coordHR[0]*tmp_l,fill_opacity=0.5,stroke_opacity=0.5,color=WHITE,fill_color=WHITE,z_index=-5).move_to(coord.get_origin()-coordHU*2*tmp_l)
        geom2 = Line(p2.get_center()-coordHU,p2.get_center()+coordHU,stroke_opacity=0.5,z_index=-5)
        geom3 = Line(p3.get_center()-coordHU,p3.get_center()+coordHU,stroke_opacity=0.5,z_index=-5)
        geometry= [geom0,geom1,geom2, geom2]
        self.add(coord,p0,p1,p2,p3,tp0,tp1,tp2,tp3,l01,l12,l23)


        # ======== copied from other scene
        tc = r'\boldsymbol{c}'
        tx = r'\boldsymbol{x}'
        tC = MathTex(r'C = \begin{bmatrix} \phantom{c_i(x_i,x_i,x_i),x_i} \\ \phantom{c_i(x_i,x_i,x_i)} \end{bmatrix} = \boldsymbol{0}').to_corner(UP+LEFT)
        tC1 = MathTex(tc,'_1 (',tx,'_0,',tx,'_1,',tx,'_2)').move_to(tC).shift(RIGHT*0.0+UP*0.3)
        tC2 = MathTex(tc,'_2 (',tx,'_1,',tx,'_2,',tx,'_3)').move_to(tC).shift(RIGHT*0.0+DOWN*0.3)
        tcdescr = VGroup(tC,tC1,tC2)
        # ======== copied from other scene

        matdc = DecimalMatrix(lDC, element_to_mobject_config={"num_decimal_places": 1}).scale(0.75)
        mathdcDesc = MathTex(r'\Delta C = ').next_to(matdc,LEFT)
        matdc_full =  VGroup(mathdcDesc, matdc).next_to(tcdescr, DOWN, aligned_edge=LEFT).shift(LEFT * 0.32)


        self.add(tcdescr)

        self.play(Create(geom0),FadeIn(geom1),Create(geom2),Create(geom3))
        self.wait(1); self.next_slide()
        self.play(Create(matdc_full))
        self.wait(1); self.next_slide()

        def localDir(dir:np.array):
            return coord.c2p(dir[0],dir[1]) - coord.c2p(0,0)


        def vis_vert(x: Vertex2d):
            v = VGroup()
            if not type(x.p) is np.ndarray:return v
            p = coord.c2p(x.p[0],x.p[1])
            v.add(Arrow(p,p+localDir(x.n)   ,stroke_width=2,buff=0,color=RED,tip_length=0.1),)
            v.add(Arrow(p,p+localDir(x.dpdu),stroke_width=2,buff=0,color=BLUE,tip_length=0.1),)
            v.add(Arrow(p+localDir(x.n),p+localDir(x.n)+localDir(x.dndu),stroke_width=2,buff=0,color=GREEN,tip_length=0.1),)
            return v

        def vis_proj_vert(x: Vertex2d):
            v = VGroup()
            if not type(x.p) is np.ndarray:return VGroup(p0)
            p = coord.c2p(x.p[0],x.p[1])
            v = VGroup()
            v.add(Dot(p,color=PURPLE))
            return v


        fullx0 = always_redraw(lambda: vis_vert(lax0))
        fullx1 = always_redraw(lambda: vis_vert(lax1))
        fullx2 = always_redraw(lambda: vis_vert(lax2))
        fullx3 = always_redraw(lambda: vis_vert(lax3))

        vgl12 = VGroup(copy.deepcopy(l01).set_color(RED).set_opacity(1),copy.deepcopy(l12).set_color(RED).set_opacity(1),z_index=2)
        vgl23 = VGroup(copy.deepcopy(l12).set_color(RED).set_opacity(1),copy.deepcopy(l23).set_color(RED).set_opacity(1),z_index=2)
        matdcr1 = SurroundingRectangle(matdc.get_rows()[0][0:3],color=RED)
        matdcr2 = SurroundingRectangle(matdc.get_rows()[1][1:4],color=RED)


        braceB0 = Brace(matdc.get_columns()[0])
        braceB0t = braceB0.get_tex('B_0')
        braceB3 = Brace(matdc.get_columns()[3])
        braceB3t = braceB3.get_tex('B_3')
        braceA = Brace(matdc.get_columns()[1:3])
        braceAt = braceA.get_tex('A')
        braceVG = VGroup(braceB0,braceB0t,braceA,braceAt,braceB3,braceB3t)

        self.play(Create(matdcr1), Create(vgl12))
        self.wait(1);self.next_slide()
        self.play(FadeOut(matdcr1),FadeOut(vgl12),Create(matdcr2),Create(vgl23))
        self.wait(1);self.next_slide()
        self.play(FadeOut(matdcr2),FadeOut(vgl23))

        self.play(FadeIn(braceVG))
        self.wait(1);self.next_slide()

        matdx1 = MathTex(r'{\partial \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \over \partial x_0 } = ' )
        matdx1.next_to(matdc_full,DOWN,DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*6,LEFT)
        matdx1x3 = MathTex(r'{\partial \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \over \partial x_3 } = ' )
        matdx1x3.next_to(matdc_full,DOWN,DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*6,LEFT)
        matdx2 = MathTex(r'-' ).next_to(matdx1)
        matA_subgroup = copy.deepcopy(VGroup(*(matdc.get_columns()[1:3])))
        matA = DecimalMatrix(lA).scale(0.75).next_to(matdx2).shift(DOWN*0.3)
        matA_desc = MathTex('-1').next_to(matA,UP+RIGHT,buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).scale(0.7)
        matinvA = DecimalMatrix(lInvA).scale(0.75).next_to(matdx2).shift(DOWN*0.3)

        matB0_subgroup = copy.deepcopy(VGroup(*(matdc.get_columns()[0])))
        matB0 = DecimalMatrix([[ lB0[0] ],[ lB0[1] ]]).scale(0.75).next_to(matA,buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2)
        matB3_subgroup = VGroup(*(matdc.get_columns()[3]))
        matB3 = DecimalMatrix([[ lB3[0] ],[ lB3[1] ]]).scale(0.75).next_to(matA, buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2)

        matlts0 = DecimalMatrix([[ lts1[0] ],[ lts1[1] ]] ).scale(0.75).next_to(matdx1).shift(DOWN*0.3)
        matlts3 = DecimalMatrix([[ lts3[0] ],[ lts3[1] ]] ).scale(0.75).next_to(matdx1x3).shift(DOWN*0.3)



        self.play(Create(matdx1))
        self.play(Transform(matA_subgroup,matA),Create(matA_desc),Create(matdx2))
        self.play(Transform(matB0_subgroup,matB0))
        self.wait(1); self.next_slide()
        self.play(ReplacementTransform(matA_subgroup,matinvA),FadeOut(matA_desc))
        tmp_vg = VGroup(matinvA,matB0);self.remove(matinvA,matB0, matB0_subgroup); self.add(tmp_vg);
        self.play(ReplacementTransform(tmp_vg,matlts0),FadeOut(matdx2))

        self.wait(1); self.next_slide()
        self.play(FadeIn(VGroup(fullx0,fullx1,fullx2,fullx3)))
        self.play(p0.animate.set_color(RED),run_time=0.5)
        self.wait(1); self.next_slide()
        self.play(u0.animate.set_value(1.0),run_time=4)
        self.wait(1); self.next_slide()
        self.play(u0.animate.set_value(-1.0),run_time=4)
        self.wait(1); self.next_slide()
        self.play(u0.animate.set_value(0),run_time=2)
        self.play(p0.animate.set_color(WHITE))
        self.next_slide()

        self.play(ReplacementTransform(matlts0,matlts3),ReplacementTransform(matdx1,matdx1x3))
        self.wait(1); self.next_slide()
        self.play(p3.animate.set_color(RED),run_time=0.5)
        self.play(u3.animate.set_value(1.0),run_time=4)
        self.wait(1); self.next_slide()
        self.play(u3.animate.set_value(-1.0),run_time=4)
        self.wait(1); self.next_slide()
        self.play(u3.animate.set_value(0),run_time=2)
        self.play(p3.animate.set_color(WHITE))
        self.next_slide()


        # TODO Create DC
        self.wait(1)

        def up_geom0(l: Line):
            coordHR = coord.c2p(0.5, 0) - coord.get_origin(); coordHU = coord.c2p(0, 0.5) - coord.get_origin()
            l.put_start_and_end_on(p0.get_center() - coordHR, p0.get_center() + coordHR)

        def create_geom1():
            coordHR = coord.c2p(0.5, 0) - coord.get_origin()
            coordHU = coord.c2p(0, 0.5) - coord.get_origin()
            return Circle(2 * coordHR[0]*tmp_l, fill_opacity=0.5, stroke_opacity=0.5, color=WHITE, fill_color=WHITE,
                   z_index=-5).move_to(coord.get_origin() - coordHU * 2*tmp_l)
        def up_geom2(l: Line):
            coordHR = coord.c2p(0.5, 0) - coord.get_origin(); coordHU = coord.c2p(0, 0.5) - coord.get_origin()
            l.put_start_and_end_on(p2.get_center()-coordHU,p2.get_center()+coordHU)
        def up_geom3(l: Line):
            coordHR = coord.c2p(0.5, 0) - coord.get_origin(); coordHU = coord.c2p(0, 0.5) - coord.get_origin()
            l.put_start_and_end_on(p3.get_center()-coordHU,p3.get_center()+coordHU)


        self.remove(geom1); geom1 = always_redraw(create_geom1); self.add(geom1)
        geom0.add_updater(up_geom0); geom2.add_updater(up_geom2); geom3.add_updater(up_geom3)
        self.play(
            coord.animate.move_to(ORIGIN+UP*3 + RIGHT*3).scale(4),
            FadeOut(tcdescr), FadeOut(matdc_full), FadeOut(matdx1x3), FadeOut(matlts3),
            FadeOut(braceVG)
        )
        geom0.remove_updater(up_geom0); geom2.remove_updater(up_geom2); geom3.remove_updater(up_geom3)

        self.play(p0.animate.set_color(RED),run_time=0.5)
        self.play(u0.animate.set_value(1.0),run_time=4)
        self.wait(1); self.next_slide()

        fullpx1 = always_redraw(lambda: vis_proj_vert(lpax1))
        fullpx0 = always_redraw(lambda: vis_proj_vert(lpax0))
        lp21 = Line(p2.get_center(),fullpx1[0].get_center(),z_index=1,stroke_opacity=1.0,color=PURPLE)
        lp10 = Line(fullpx1[0].get_center(),fullpx0[0].get_center(),z_index=-1,stroke_opacity=1.0,color=PURPLE)
        lp21.add_updater(lambda l: l.put_start_and_end_on(p2.get_center(),fullpx1[0].get_center()))
        lp10.add_updater(lambda l: l.put_start_and_end_on(fullpx1[0].get_center(),fullpx0[0].get_center()))
        lpvg = VGroup(lp21,fullpx1,lp10,fullpx0)
        self.play(Create(lpvg),FadeOut(fullx1))
        self.wait(1); self.next_slide()

        self.wait(1); self.next_slide()
        self.play(u0.animate.set_value(0.0),run_time=1.0)
        geom0.add_updater(up_geom0); geom2.add_updater(up_geom2); geom3.add_updater(up_geom3)
        self.play(coord.animate.scale(1/2).move_to(ORIGIN),run_time=1.0)
        geom0.remove_updater(up_geom0); geom2.remove_updater(up_geom2); geom3.remove_updater(up_geom3)
        target_p = Dot(coord.c2p(lx0.p[0]+0.5,lx0.p[1]),z_index=15,color=GREEN)
        self.play(Create(target_p))
        self.wait(1);self.next_slide()
        self.play(u0.animate.set_value(-0.3),run_time=2.0)
        self.wait(1); self.next_slide()
        self.play(u0.animate.set_value(-0.5),run_time=2.0)
        self.wait(1); self.next_slide()
        self.play(u0.animate.set_value(-0.58),lp21.animate.set_color(GREEN),lp10.animate.set_color(GREEN),run_time=2.0,)


        self.wait(1); self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()






    def construct(self):
        self.cost_function()
        self.create_c_der()
        # self.tangent_space()
        # Explain Implicit Function Theorem
        # Explain Variables
        # Explain Cost Function
        # Explain Cost Function Derivative
        # Explain Cost Function Derivative solved for B1 or B3

mx0 = Vertex(
    np.array([-1.0, 0.0, 2.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
    np.array([0.0,  0.0,-1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
    1.0, None, None, None
)

mx1 = Vertex(
    np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
    1.0, None, None, None
)

mx2 = Vertex(
    np.array([2.0, 0.0, 3.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
    1.0, None, None, None
)

class p01_1(Slide):
    scale = 1
    def construct(self):
        # TODO INTRO

        doub_buf = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2
        coord = Axes(x_range=[-1.4, 1.4, 1],x_length=5*self.scale, y_range=[-1.4, 1.4, 1], y_length=5*self.scale,axis_config={"include_numbers": False}).to_edge(RIGHT)
        self.add(coord)
        c = Circle((coord.c2p(1,0)-coord.c2p(0,0))[0],WHITE).move_to(coord.get_origin())
        texImplFunc= Tex(r'\underline{Implicit Function Theorem}',font_size=1.5*DEFAULT_FONT_SIZE).to_corner(UP+LEFT)
        texf= Tex('given constraint ').next_to(texImplFunc,DOWN,doub_buf,LEFT)
        texf2= MathTex(r'c({{ {x} }},{{ {y} }}) = {{ {x} }}^2 + {{ {y} }}^2 -1 = 0').next_to(texf,RIGHT)
        texf2.set_color_by_tex("{x}", RED).set_color_by_tex("{y}", GREEN)

        texDf= MathTex(r'(Dc)(a,b) = \begin{bmatrix} 2a & 2b \end{bmatrix}').next_to(texf,DOWN,doub_buf,aligned_edge=LEFT)
        texIf= MathTex(r'\text{If right site ($2b$) is invertible:}').next_to(texDf,DOWN,doub_buf,aligned_edge=LEFT)
        texIf2= MathTex(r'\text{there exist } g \text{ s.t. } g({{a}}) = b ').next_to(texIf,DOWN,aligned_edge=LEFT)
        texIf3= MathTex(r'\text{and s.t. } c({{ {x} }}, g({{ {y} }})) = 0 \text{ in neighboorhood } ').next_to(texIf2,DOWN,aligned_edge=LEFT)
        texIf3.set_color_by_tex("{x}", RED).set_color_by_tex("{y}", GREEN)



        texdescr = MathTex(r'\text{calc } {d{{ {y} }} \over d{{ {x} }}} \text{: }').next_to(texIf3,DOWN,doub_buf,aligned_edge=LEFT)
        texdescr.set_color_by_tex("{x}", RED).set_color_by_tex("{y}", GREEN)
        MathTex.set_default(substrings_to_isolate=["x","y"])
        texDf2= MathTex(r'x^2 + y^2 = 1',substrings_to_isolate=["x","y"]).next_to(texdescr)
        texDf3= MathTex(r'{d \over dx }(x^2 + y^2) = {d \over dx} 1').next_to(texdescr)
        texDf4= MathTex(r'{d \over dx }(x^2) + {d \over dx}( y^2) = {d \over dx} 1').next_to(texdescr)
        texDf5= MathTex(r'2x + 2y \cdot {dy \over dx} = 0').next_to(texdescr)
        texDf6= MathTex(r'2y  \cdot {dy \over dx} = -2x').next_to(texdescr)
        MathTex.set_default(substrings_to_isolate=None)
        texDf7= MathTex(r'{d{{y}} \over d{{x}}} {{=}} - {{{x}} \over {{y}}}').next_to(texdescr)
        for t in [texDf2,texDf3,texDf4,texDf5,texDf6,texDf7]:
            t.set_color_by_tex("x",RED).set_color_by_tex("y",GREEN)

        t = ValueTracker(math.pi*1/16)
        def up_p(d: Dot):
            d.move_to(coord.c2p(math.sin(t.get_value()),math.cos(t.get_value())))
        p = Dot()
        up_p(p)
        p.add_updater(lambda d: up_p(d))

        def up_der(a: Arrow):
            pcoord = coord.p2c(p.get_center())
            dirx = 1.0*pcoord[1]
            diry = - (pcoord[0]/pcoord[1]) * dirx
            newp = pcoord + np.array([dirx,diry]) #/np.linalg.norm(np.array([dirx,diry]))
            a.put_start_and_end_on(coord.c2p(pcoord[0],pcoord[1]),coord.c2p(newp[0],newp[1]))
        D = Arrow(buff=0,color=RED)
        up_der(D)
        D.add_updater(lambda a: up_der(a))

        self.play(Write(texImplFunc),Create(coord),run_time=1)
        self.wait(1);self.next_slide()
        self.play(Write(texf),Write(texf2),Create(c))
        self.wait(1);self.next_slide()
        self.play(Create(texDf))
        self.wait(1);self.next_slide()
        self.play(Create(texIf),Create(texIf2))
        self.wait(1);self.next_slide()
        self.play(Create(texIf3))
        self.wait(1);self.next_slide()

        self.play(Create(VGroup(texdescr,texDf2)))
        self.wait(1);self.next_slide()
        self.play(TransformMatchingShapes(texDf2,texDf3),run_time=0.5); self.wait(0.5)
        self.play(TransformMatchingShapes(texDf3,texDf4),run_time=0.5); self.wait(0.5)
        self.play(TransformMatchingShapes(texDf4,texDf5),run_time=0.5); self.wait(0.5)
        self.play(TransformMatchingShapes(texDf5,texDf6),run_time=0.5); self.wait(0.5)
        self.play(TransformMatchingShapes(texDf6,texDf7),run_time=0.5); self.wait(0.5)

        self.play(Create(p),Create(D))
        self.play(t.animate.set_value(7/16*math.pi),run_time=5)

        self.wait(1); self.next_slide()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.clear()


class p01_2(ThreeDSlide):
    def construct(self):
        sw = DEFAULT_STROKE_WIDTH
        sw = np.array([DEFAULT_STROKE_WIDTH])
        Arrow3D.set_default(resolution=8,stroke_width=sw)
        Dot3D.set_default(resolution=8)
        # MathTex.set_default(resolution=2, stroke_with=[sw])
        axes = ThreeDAxes().scale(1.25)
        axes.x_axis.rotate(-90*DEGREES,X_AXIS)
        self.set_camera_orientation(45*DEGREES,-45*DEGREES)
        cyl = Cylinder(height=10,direction=Y_AXIS,stroke_width=sw,resolution=(8,8),stroke_color=GRAY,fill_color=BLACK,fill_opacity=0,stroke_opacity=0.5,show_ends=False)
        self.add(axes,cyl)
        x = mx1
        p = Dot3D(x.p)
        tx = r'\boldsymbol{x}'
        tn = r'\boldsymbol{n}'
        texp = MathTex(tx).next_to(p,LEFT)
        ttp = MathTex(tx, '= (0,1,0)').to_corner(UL)
        dpdu = Arrow3D(x.p,x.p+x.dpdu,color=BLUE)
        tdpdu  = MathTex(r'\partial_u',tx,color=dpdu.color).next_to(dpdu)
        ttdpdu  = MathTex(r'\partial_u',tx,' = (1,0,0)',color=dpdu.color).next_to(ttp,DOWN,aligned_edge=LEFT)
        dpdv = Arrow3D(p.get_center(),x.p+x.dpdv,color=dpdu.color)
        tdpdv = MathTex(r'\partial_v ',tx,color=dpdv.color).next_to(dpdv.get_end(),UP)
        ttdpdv  = MathTex(r'\partial_v',tx,' = (0,0,1)',color=dpdu.color).next_to(ttdpdu,DOWN,aligned_edge=LEFT)
        n = Arrow3D(p.get_center(),x.p+x.n,color=RED)
        texn =  MathTex(tn, color=n.color).next_to(n.get_end(),LEFT)
        ttn = MathTex(tn, '= (0,1,0)',color=n.color).next_to(ttdpdv,DOWN,aligned_edge=LEFT)
        dndu = Arrow3D(x.p+x.n,x.p+x.n+x.dndu,color=GREEN)
        tdndu  = MathTex(r'\partial_u',tn,color=dndu.color).next_to(dndu)
        ttdndu  = MathTex(r'\partial_u',tn,' = (1,0,0)',color=dndu.color).next_to(ttn,DOWN,aligned_edge=LEFT)
        dndv = Dot3D(x.p+x.n,color=dndu.color)
        tdndv  = MathTex(r'\partial_v',tn,color=dndu.color).next_to(dndv,LEFT)
        ttdndv  = MathTex(r'\partial_v',tn,' = (0,0,0)',color=dndu.color).next_to(ttdndu,DOWN,aligned_edge=LEFT)

        coord = Axes(x_range=[-3.5, 3.5, 1], y_range=[-0.5, 3.5, 1],axis_config={"include_numbers": True}).rotate(90*DEGREES,X_AXIS)
        geom1 = Circle(fill_opacity=0.5,stroke_opacity=0.5,color=WHITE,fill_color=WHITE).rotate(90*DEGREES,X_AXIS)

        self.wait(1); self.next_slide()
        self.add_fixed_in_frame_mobjects(ttp)
        self.play(Create(p),FadeIn(texp))
        self.wait(1); self.next_slide()
        self.add_fixed_in_frame_mobjects(ttdpdu,ttdpdv)
        self.play(Create(dpdu),FadeIn(tdpdu),Create(dpdv),FadeIn(tdpdv))
        self.wait(1); self.next_slide()
        self.add_fixed_in_frame_mobjects(ttn)
        self.play(Create(n),FadeIn(texn))
        self.wait(1); self.next_slide()
        self.add_fixed_in_frame_mobjects(ttdndu,ttdndv)
        self.play(Create(dndu),FadeIn(tdndu),Create(dndv),FadeIn(tdndv))
        self.wait(1); self.next_slide()
        # self.add(p,texp,dpdu,tdpdu,dpdv,tdpdv,n,texn,dndu,tdndu,dndv,tdndv)
        # self.wait(1)
        def anRot(o):
            return o.animate.rotate(90*DEGREES,X_AXIS)
        self.remove(dndv)
        self.move_camera(90*DEGREES,-90*DEGREES,added_anims=[
            FadeOut(axes.y_axis), anRot(texp), anRot(texn), anRot(tdpdu), FadeOut(tdpdv), anRot(tdndu), FadeOut(tdndv),
            FadeOut(ttdpdv), FadeOut(ttdndv),
            ttn.animate.next_to(ttdpdu,DOWN,aligned_edge=LEFT),
            ttdndu.animate.next_to(ttdpdu,DOWN,buff=ttp.get_y()-ttdpdu.get_y()+DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,aligned_edge=LEFT),
            FadeOut(cyl),FadeIn(geom1), FadeOut(dpdv)
        ])
        # self.play(Create(cyl))
        self.wait(1)
