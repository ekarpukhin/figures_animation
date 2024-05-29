import taichi as ti
import taichi_glsl as ts
from base_shader import *


red = ts.vec3(1., 0., 0.)
green = ts.vec3(0., 1., 0.)
blue = ts.vec3(0., 0., 1.)
black = ts.vec3(0.)
white = ts.vec3(1.)
dur = 5
fig_size = 0.05
speed = 0.1

@ti.func
def sd_circle(p, r):
    return p.norm() - r

@ti.func
def box(p, b):
    d = abs(p)-b
    return max(d,0.0).norm() + min(max(d[0],d[1]), 0.0)

@ti.func
def rot(a):
    """
    Функция для расчета матрицы поворота.
    При помощи декоратора ti.func может выполняться на видеокарте.
    :param a: угол поворота в радианах
    :return: матрица поворота на угол `a` в двумерном пространстве
    """
    c = ti.cos(a)
    s = ti.sin(a)
    return ts.mat([c, -s], [s, c])


@ti.func
def clamp(x, low, high):
    """
    Обрезание точки - если вне интервала, то возвращается ближайший конец, иначе сама точка
    При помощи декоратора ti.func может выполняться на видеокарте.
    :param x: точка vec1
    :param low, high: начало и конец отрезка
    :return: число (одномерная точка)
    """
    return ts.min(ts.max(x, high), low)

@ti.func
def sd_segment(p, a, b):
    """
    sdf-функция для отрезка.
    При помощи декоратора ti.func может выполняться на видеокарте.
    :param p: точка на плоскости, vec2
    :param a, b: начало и конец отрезка, vec2
    :return: sdf-функция для отрезка
    """
    pa = p - a
    ba = b - a
    h = ts.clamp((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return (pa - ba * h).norm()



@ti.dataclass
class Fig:
    pos: ti.math.vec2
    sdf: ti.i32
    index: ti.i32
    start_t: ti.f32
    # omega: ti.f32
    color: ti.math.vec3
    @ti.func
    def growing_box(self, t, p, omega=0.5, b=ts.vec2(fig_size, fig_size)):
        return box(rot(omega*(t-self.start_t)) @ p, b + (t-self.start_t)*speed*ts.vec2(1, 1))

    @ti.func
    def growing_circle(self, t, p, r = fig_size):
        return sd_circle(p, r + speed*(t-self.start_t))

    @ti.func
    def growing_segment(self, t, p, omega=0.5, a = ts.vec2(0,0), b = fig_size*ts.vec2(1,1)):
        return sd_segment(rot(omega*(t-self.start_t)) @ p, a, b + (t-self.start_t)*speed*ts.vec2(1, 1)) - 0.02

    @ti.func
    def curr_color(self,d,t):
        return self.color * ti.exp(-50*ti.abs(d)) * (-((2*((t-self.start_t)/dur)-1)**2) + 1)

    @ti.func
    def signed_distance(self, uv, t) -> ti.f32:
        d = 0.
        if(self.sdf==0):
            d = self.growing_circle(t, uv - self.pos)
        if(self.sdf==1):
            d = self.growing_box(t, uv - self.pos)
        if(self.sdf==2):
            d = self.growing_segment(t,uv-self.pos)
        if t < self.start_t or t > self.start_t + dur:
            d = 1.
        return d


class Shader(BaseShader):

    def __init__(self,
                 title: str,
                 res: tuple[int, int] | None = None,
                 gamma: float = 2.2,
                 count: int = 7
                 ):
        super().__init__(title, res=res, gamma=gamma)
        self.figures = Fig.field(shape=(count,))
        self.count = count

        self.figures[0] = Fig(-ts.vec2(-0.1, -0.1), 0, 0, 0, red)
        self.figures[1] = Fig(-ts.vec2(0.1, 0.1), 1, 0, 1, blue)
        self.figures[2] = Fig(-ts.vec2(0.3, 0.2), 1, 0, 2, green)
        self.figures[3] = Fig(-ts.vec2(0.2, -0.3), 2, 0, 4, blue)
        self.figures[4] = Fig(-ts.vec2(0.1, -0.3), 0, 0, 3, white)
        self.figures[5] = Fig(-ts.vec2(0.0, 0.2), 1, 0, 4, red)
        self.figures[6] = Fig(-ts.vec2(-0.2, 0.3), 2, 0, 4, blue)

    @ti.func
    def main_image(self, uv, t):
        prev_color = black
        d = -1.
        for i in range(self.count):
            # cur_d = self.circles[i].signed_distance(uv)
            curr_d = self.figures[i].signed_distance(uv, t)
            if curr_d <= 0:
                color = self.figures[i].curr_color(curr_d, t)
                prev_color = ts.mix(color, prev_color, curr_d/d)
                d = max(d, curr_d)   

        return prev_color


if __name__ == "__main__":

    ti.init(arch=ti.gpu)

    shader = Shader(
        "Figures",
        gamma=1.0
    )

    shader.main_loop()
