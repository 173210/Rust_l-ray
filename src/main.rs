extern crate l_ray;
extern crate rand;
extern crate threadpool;

use std::f64::consts::*;
use std::fs::File;
use std::io::*;
// PIに使った

use rand::random;

use l_ray::source::{ vector::*, *};

// 画像の情報
const WIDTH: usize = 1200;
const HEIGHT: usize = 800;
const MAX: usize = 255;

// pixelあたりのサンプル数（よく解ってない）
const SPP: u32 = 10;

// スレッドの数
const WORKS: usize = 10;

fn main() {
	let wid = WIDTH as f64;
	let hei = HEIGHT as f64;

	let up: V = V::new_tri(0.0, 1.0, 0.0);
	// "上"の方向
	let fov: f64 = 30.0 * PI / 180.0;
	let tf = (fov * 0.5).tan();
	// 視野角 ラジアンに直した
	let aspect: f64 = wid / hei;
	// アスペクト比（4:3みたいなやつ）

	// 球体のみ用
	// let eye: V = V::new_sig(5.0);
	// // 目の位置
	// let center: V = V::new_sig(0.0);
	// // 注視点

	// in room用
	let eye = V::new_tri(50.0, 52.0, 295.6);
	let center = eye + V {
		x: 0.0,
		y: -0.042612,
		z: -1.0,
	};

	// 視線の基底（全部垂直な単位ベクトル）
	let WE: V = V::norm(eye - center);
	// 視線の単位ベクトルj
	let UE: V = V::norm(V::cross(up, WE));
	// 視線と"上"の方向に垂直な単位ベクトル
	let VE: V = V::cross(WE, UE);
	// WEとUEに垂直な単位ベクトル（双方正規化されてるため正規化の必要なし）

	let tonemap = |v: f64| {
		use std::cmp::*;
		min(max((v.powf(1.0 / 2.2) * 255.0) as u32, 0), 255)
		// ちゃんとガンマ補正したよ
	};

	// Threads
	let mut threads: Vec<std::thread::JoinHandle<Vec<[u32; 3]>>> = Vec::with_capacity(WORKS);

	// scene 初期化
	//let scene = obj::Scene::new_mul();
	let scene = obj::Scene::in_room();

	for work in (0..WORKS).rev() {
		let scene = scene.clone();

		threads.push(std::thread::spawn(move || {
			let top = HEIGHT * work / WORKS;
			let bottom = HEIGHT * (work + 1) / WORKS;
			let mut ray = obj::Ray::new();
			let mut write_v: Vec<[u32; 3]> = Vec::with_capacity((bottom - top) * WIDTH);

			for y in (top..bottom).rev() {
				for x in 0..WIDTH {
					let mut write_push = V::new();

					for j in 0..SPP as usize {
						let random_f64 = random::<f64>();

						// rayの値
						ray.o = eye;
						// 目の位置
						{
							let rpx = 2.0 * (x as f64 + random_f64) / wid - 1.0;
							let rpy = 2.0 * (y as f64 + random_f64) / hei - 1.0;
							let w: V = V::norm(V {
								x: aspect * tf * rpx,
								y: tf * rpy,
								z: -1.0,
							});
							ray.d = UE * V::new_sig(w.x) + VE * V::new_sig(w.y) + WE * V::new_sig(w.z);
						// 目線の向き
						}

						let mut ill_l = V::new_sig(0.0);
						let mut refl_l = V::new_sig(1.0) ;

						for depth in 0..10 {
							let h: Option<obj::Hit> = scene.intersect(&ray, 1e-4, 1e+10);

							if let Some(s) = h {
							// 球にレイが当たった時

								// 光度更新
								ill_l = ill_l + refl_l * s.sphere.ill;

								// 球上の交点をレイの原点にする
								ray.o = s.p;
								// レイの方向
								ray.d = {
									// nが法線、u,vがそれに直交するベクトル。
									// nから直交単位ベクトルを生成してる
       								let n = if V::dot(s.n, -ray.d) > 0.0 { s.n } else { -s.n };
									let (u, v) = tangent_space(n);
									let d: V = {
										let r = random::<f64>();
										let r_sqrt = r.sqrt();
										let t = 2.0 * PI * random_f64;
										V {
											x: r_sqrt * t.cos(),
											y: r_sqrt * t.sin(),
											z: 0.0_f64.max(1.0 - r).sqrt(),
										}
									};
									u * V::new_sig(d.x) + v * V::new_sig(d.y) + n * V::new_sig(d.z)
								};
								refl_l = refl_l * s.sphere.refl;
							} else {
							// 当たらなかったらループから抜ける
								break;
							}

							if refl_l.x == 0.0 && refl_l.y == 0.0 && refl_l.z == 0.0 {
							// if ill_l * ill_l.y * ill_l.z == 0.0 {
								break;
							}
						}

						write_push = write_push + ill_l;
					}

					write_push = write_push / V::new_sig(SPP as f64);
					write_v.push([tonemap(write_push.x), tonemap(write_push.y), tonemap(write_push.z)]);
				}
			}

			write_v
		}));
	}

	// ppmファイル生成
	let mut file = File::create("ideal.ppm").unwrap();
	file.write_all(format!("P3\n{} {}\n{}\n", WIDTH, HEIGHT, 255).as_bytes())
		.unwrap();

	for thread in threads {
		for n in thread.join().unwrap() {
			file.write_all(format!("{} {} {}\n", n[0], n[1], n[2]).as_bytes()).unwrap();
		}
	}
}

fn tangent_space(n: V) -> (V, V){
	// 一つのベクトルを元に、直交の単位ベクトルを生成する関数。
	// 外積を使うことで同じ事はできるが、こちらの方が速度で勝っている。
	// 理解が難しいので今後の課題
	let s = if n.z >= 0.0 { 1.0 } else { -1.0 };

	let a = -1.0 / (s + n.z);
	let b = n.x * n.y * a;

	// return
	(
		V {
			x: 1.0 + s * n.x * n.x * a,
			y: s * b,
			z: -s * n.x,
		},
		V {
			x: b,
			y: s + n.y * n.y * a,
			z: -n.y,
		}
	)
}
