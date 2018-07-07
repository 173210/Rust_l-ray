use source::*;

#[derive(Debug, Clone)]
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

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Sphere {
    pub p: vector::V,
    // 中心位置
    pub r: f64,
    // 半径
    pub refl: vector::V,
    // 反射率
    pub ill: vector::V,
    // 光度
}

impl Sphere {
    pub fn new() -> Sphere {
        Sphere {
            p: vector::V::new(),
            r: 0.0,
            refl: vector::V::new(),
            ill: vector::V::new(),
        }
    }
    pub fn from(p: vector::V, r: f64, refl: vector::V, ill: vector::V) -> Sphere {
        Sphere {
            p: p,
            r: r,
            refl: refl,
            ill: ill,
        }
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
            hit.sphere = self;
            return Some(hit);
        }

        let t2 = b + det.sqrt();
        if tmin < t2 && t2 < tmax {
            hit.t = t2;
            hit.sphere = self;
            Some(hit)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
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
                Sphere::from(
                    vector::V {
                        x: -0.2,
                        y: 0.0,
                        z: 0.0,
                    },
                    0.5,
                    vector::V {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    vector::V {
                        x: 0.2,
                        y: 0.0,
                        z: 0.0,
                    },
                    0.5,
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
            ],
        }
    }
    pub fn in_room() -> Scene {
        Scene {
            spheres: vec![
                Sphere::from(
                    //左の壁
                    vector::V {
                        x: 1e5 + 1.0,
                        y: 40.8,
                        z: 81.6,
                    },
                    1e5,
                    vector::V {
                        x: 0.3,
                        y: 0.7,
                        z: 0.3,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    //右の壁
                    vector::V {
                        x: -1e5 + 99.0,
                        y: 40.8,
                        z: 81.6,
                    },
                    1e5,
                    vector::V {
                        x: 0.3,
                        y: 0.3,
                        z: 0.7,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    //奥の壁
                    vector::V {
                        x: 51.0,
                        y: 40.8,
                        z: 1e5,
                    },
                    1e5,
                    vector::V {
                        x: 0.75,
                        y: 0.75,
                        z: 0.75,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ), //
                Sphere::from(
                    //床
                    vector::V {
                        x: 51.0,
                        y: 1e5,
                        z: 81.6,
                    },
                    1e5,
                    vector::V {
                        x: 0.75,
                        y: 0.75,
                        z: 0.75,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    //天井
                    vector::V {
                        x: 51.0,
                        y: -1e5 + 81.6,
                        z: 81.6,
                    },
                    1e5,
                    vector::V {
                        x: 0.75,
                        y: 0.75,
                        z: 0.75,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    vector::V {
                        x: 27.0,
                        y: 16.5,
                        z: 47.0,
                    },
                    16.5,
                    vector::V {
                        x: 0.95,
                        y: 0.95,
                        z: 0.95,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    vector::V {
                        x: 73.0,
                        y: 16.5,
                        z: 78.0,
                    },
                    16.5,
                    vector::V {
                        x: 0.7,
                        y: 0.9,
                        z: 0.9,
                    },
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                Sphere::from(
                    // 上の照明
                    vector::V {
                        x: 51.0,
                        y: 681.6,
                        z: 81.6,
                    },
                    601.0,
                    vector::V {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    vector::V {
                        x: 12.0,
                        y: 12.0,
                        z: 12.0,
                    },
                ),
            ],
        }
    }
    pub fn intersect(self: &Scene, ray: &Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut minh: Option<Hit> = None;
        let mut s: Sphere = Sphere::new();
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
            sphere: Sphere::new(),
        }
    }
}
