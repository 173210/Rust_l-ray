use source::*;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Hit {
    pub t: f64,
    // レイの原点-交差点の距離
    pub p: vector::V,
    // 交点の位置ヴェクトル
    pub n: vector::V,
    // 交点の法線:w
    pub sphere: Sphere,
    // 当たった球の情報
}

impl Hit {
    fn new() -> Hit {
        Hit {
            t: 0.0,
            p: vector::V::new(),
            n: vector::V::new(),
            sphere: Sphere::new(vector::V::new(), 0.0),
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Sphere {
    pub p: vector::V,
    // 中心位置
    pub r: f64,
    // 半径
}

impl Sphere {
    pub fn new(p: vector::V, r: f64) -> Sphere {
        Sphere { p: p, r: r }
    }

    pub fn intersect(self: Sphere, ray: &Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let op = self.p - ray.o;
        let b = vector::V::dot(op, ray.d);
        let det = b * b - vector::V::dot(op, op) + self.r * self.r;
        // 判別式

        let mut hit = Hit::new();

        if det < 0.0 {
            // 実数解を持たない
            return None;
        }

        let t1 = b - det.sqrt();
        if tmin < t1 && t1 < tmax {
            hit.t = t1;
            return Some(hit);
        }

        let t2 = b + det.sqrt();
        if tmin < t2 && t2 < tmax {
            hit.t = t2;
            Some(hit)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
}

impl Scene {
    pub fn new(s: Sphere) -> Scene {
        // Vecの意味無いね、いいよね
        Scene { spheres: vec![s] }
    }
    pub fn new_mul() -> Scene {
        Scene {
            spheres: vec![
                Sphere::new(
                    vector::V {
                        x: -0.2,
                        y: 0.0,
                        z: 0.0,
                    },
                    0.5,
                ),
                Sphere::new(
                    vector::V {
                        x: 0.2,
                        y: 0.0,
                        z: 0.0,
                    },
                    0.5,
                ),
            ],
        }
    }

    pub fn in_room() -> Scene {
        Scene {
            spheres: vec![
                Sphere::new(
                    // 左の壁
                    vector::V {
                        x: 1e5 + 1.0,
                        y: 40.8,
                        z: 81.6,
                    },
                    1e5,
                ),
                Sphere::new(
                    // 右の壁
                    vector::V {
                        x: -1e5 + 99.0,
                        y: 40.8,
                        z: 81.6,
                    },
                    1e5,
                ),
                Sphere::new(
                    // 奥の壁
                    vector::V {
                        x: 50.0,
                        y: 40.8,
                        z: 1e5,
                    },
                    1e5,
                ),
                // Sphere::new(
                //     // 手前の壁（いらんやろ）
                //     vector::V {
                //         x: 50.0,
                //         y: 40.8,
                //         z: -1e5 + 170.0,
                //     },
                //     1e5,
                // ),
                Sphere::new(
                    // 床
                    vector::V {
                        x: 50.0,
                        y: 1e5,
                        z: 81.6,
                    },
                    1e5,
                ),
                Sphere::new(
                    // 天井
                    vector::V {
                        x: 50.0,
                        y: -1e5 + 81.6,
                        z: 81.6,
                    },
                    1e5,
                ),
                Sphere::new(
                    vector::V {
                        x: 27.0,
                        y: 16.5,
                        z: 47.0,
                    },
                    16.5,
                ),
                Sphere::new(
                    vector::V {
                        x: 73.0,
                        y: 16.5,
                        z: 78.0,
                    },
                    16.5,
                ),
                Sphere::new(
                    vector::V {
                        x: 50.0,
                        y: 681.6,
                        z: 81.6,
                    },
                    600.0,
                ),
            ],
        }
    }

    pub fn intersect(self: &Scene, ray: &Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut minh: Option<Hit> = None;
        let mut s: Sphere = Sphere::new(vector::V::new(), 0.0);
        let mut buf = tmax;

        for c in self.spheres.iter() {
            let h = c.intersect(ray, tmin, tmax);
            if let Some(i) = h {
                if i.t < buf {
                    buf = i.t;
                    minh = h;
                    s = *c;
                }
            }
        }

        if let Some(mut m) = minh {
            let t = vector::V::new_sig(m.t);
            m.p = ray.o + ray.d * t;
            // 交点の原点からのヴェクトル
            let r = vector::V::new_sig(s.r);
            m.n = (m.p - s.p) / r;
            // (交点までのヴェクトル - 球の中心までのヴェクトル)
            Some(m)
        } else {
            None
        }
    }
}

pub struct Ray {
    pub o: vector::V,
    // 原点
    pub d: vector::V,
    // 方向
}
impl Ray {
    pub fn new() -> Ray {
        Ray {
            o: vector::V::new(),
            d: vector::V::new(),
        }
    }
}