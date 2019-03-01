extern crate rand;
extern crate rayon;

use rayon::prelude::*;

use std::str::FromStr;
use std::io::{Write, BufWriter};
use std::fs;

use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::ops::Rem;

#[derive(Debug, PartialEq, Clone, Copy)]
struct V(f64, f64, f64); // Fields cannot have default value atm: https://github.com/rust-lang/rfcs/pull/1806

impl Add for V {
  type Output = V;
  fn add(self, rhs: Self) -> Self {
    V(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
  }
}

#[test]
fn test_add() {
  let v = V(1.0, 0.0, 1.0);
  let w = V(0.0, 1.0, 0.0);
  assert_eq!(v + w, V(1.0, 1.0, 1.0));
}

impl Mul<f64> for V {
  type Output = V;
  fn mul(self, rhs: f64) -> Self {
    V(self.0 * rhs, self.1 * rhs, self.2 * rhs)
  }
}

#[test]
fn test_mul() {
  let v = V(1.0, 1.0, 1.0);
  assert_eq!(v * 3.0, V(3.0, 3.0, 3.0));
}

impl Sub for V {
  type Output = V;
  fn sub(self, rhs: Self) -> Self { self + (rhs * -1.0)}
}

#[test]
fn test_sub() {
  let v = V(3.33, 5.0, 8.0);
  let w = V(2.33, 4.0, 7.0);
  assert_eq!(v - w, V(1.0, 1.0, 1.0)); // !?
}

impl Rem for V {
  type Output = V;
  fn rem(self, rhs: Self) -> Self {
    V(
      self.1 * rhs.2 - self.2 * rhs.1,
      self.2 * rhs.0 - self.0 * rhs.2,
      self.0 * rhs.1 - self.1 * rhs.0
    )
  }
}

#[test]
fn test_rem() {
  let v = V(1.0, 2.0, 3.0);
  let w = V(1.0, 5.0, 7.0);
  assert_eq!(v % w, V(-1.0, -4.0, 3.0));
}

impl V {
  pub fn mult(self, w: V) -> V {
    V(self.0 * w.0, self.1 * w.1, self.2 * w.2)
  }
  pub fn dot(self, w: V) -> f64 {
    self.0 * w.0 + self.1 * w.1 + self.2 * w.2
  }
  pub fn norm(self) -> V {
    let n = 1.0 / (&self.0 * &self.0 + &self.1 * &self.1 + &self.2 * &self.2).sqrt();
    self * n
  }
}

#[test]
fn test_mult() {
  let v = V(2.0, 2.0, 2.0);
  let w = V(1.0, 2.0, 3.0);
  assert_eq!(v.mult(w), V(2.0, 4.0, 6.0));
}
#[test]
fn test_dot() {
  let v = V(2.0, 2.0, 2.0);
  let w = V(1.0, 2.0, 3.0);
  assert_eq!(v.dot(w), 12.0);
}
#[test]
fn test_norm() {
  let v = V(2.0, 2.0, 2.0);
  let w = V(73.0, 16.5, 78.0);
  assert_eq!(v.norm(), V(0.5773502691896258, 0.5773502691896258, 0.5773502691896258));
  assert_eq!(w.norm(), V(0.6753110498232596, 0.15263879893265456, 0.7215652313180034));
}

struct Ray { o: V, d: V }

// Thanks to http://yajamon.hatenablog.com/entry/2018/01/06/222106
#[derive(Debug, Clone, Copy)]
enum Reflection { DIFF, SPEC, REFR }

#[derive(Debug, Clone, Copy)]
struct Sphere {
  rad: f64,
  p: V, e: V, c: V,
  refl: Reflection,
}
impl Sphere {
  pub fn intersect(self, r: &Ray) -> f64 {
    let op = self.p - r.o;
    let b = op.dot(r.d);
    let det_sqrd = b * b - op.dot(op) + self.rad * self.rad;
    if det_sqrd < 0.0 {
        return std::f64::INFINITY;
    }

    let det = det_sqrd.sqrt();
    let t1 = b - det;
    let t2 = b + det;
    if t1 > 0.0 {
        t1
    } else if t2 > 0.0 {
        t2
    } else {
        std::f64::INFINITY
    }
  }
}

fn clamp(x: f64) -> f64 {
  x.min(1.0).max(0.0)
}
#[test]
fn test_clamp() {
  assert_eq!(clamp(1.4), 1.0);
  assert_eq!(clamp(-0.5), 0.0);
  assert_eq!(clamp(0.31416), 0.31416);
}

fn to_int(x: f64) -> u8 {
  (f64::powf(clamp(x), 1.0 / 2.2) * 255.0 + 0.5) as u8
}

fn intersect(r: &Ray, spheres: &[Sphere]) -> Option<(u32, f64)> {
  //*
  let ids = 0..;
  match ids.zip(spheres.iter())
    .map(|t| (t.0, t.1.intersect(r)))
    .filter(|t| t.1 < 1e20)
    .min_by(|t, u| t.1.partial_cmp(&u.1).unwrap())
  {
    Some(tpl) => Some(tpl),
    _ => None,
  }
  // */
  /*
  let mut res = (0, std::f64::INFINITY);
  for s in 0..spheres.len() {
    let t = spheres[s].intersect(r);
    let (_, prev_t) = res;
    if t < prev_t && t > 1e-4 {
      res = (s, t);
    }
  }

  let (index, t) = res;
  if t == std::f64::INFINITY {
    None
  } else {
    Some((index as u32, t))
  }
   */
}

fn radiance(r: &Ray, depth: i32, xi: (u16, u16, u16), spheres: &[Sphere]) -> V {
  let est_radiance = match intersect(r, spheres) {
    None => V(0.0, 0.0, 0.0),
    Some((index, t)) => {
      let obj = spheres[index as usize];
      let x = r.o + r.d * t;
      let n = (x - obj.p).norm();
      let nl = if n.dot(r.d) < 0.0 { n } else { n * -1.0};
      let mut f = obj.c;

      let p = 0.299 * f.0 + 0.587 * f.1 + 0.114 * f.2;

      if depth > 5 {
        if rand::random::<f64>() < p && depth < 100 {
          f = f * (1.0 / p)
        } else {
          return obj.e;
        }
      }

      let ir = match obj.refl {
        Reflection::DIFF => {
          let r1 = 2.0 * std::f64::consts::PI * rand::random::<f64>();
          let r2 = rand::random::<f64>();
          let r2s = r2.sqrt();
          let wup = if nl.0.abs() > 0.1 { V(0.0, 1.0, 0.0) } else { V(1.0, 0.0, 0.0) } % nl;
          let tang = (nl % wup).norm();
          let bitang = (nl % tang).norm();
          let d = tang * r1.cos() * r2s + bitang * r1.sin() * r2s + nl * (1.0 - r2).sqrt();
          radiance(&Ray { o: x, d: d.norm() }, depth + 1, xi, spheres)
        },
        Reflection::SPEC => {
          let ray = Ray { o: x, d: r.d - nl * 2.0 * nl.dot(r.d) };
          radiance(&ray, depth + 1, xi, spheres)
        },
        _ => {
          let reflection = Ray { o: x, d: r.d - nl * 2.0 * nl.dot(r.d) };
          let into = n.dot(nl) > 0.0;
          let nc = 1.0;
          let nt = 1.5;
          let nnt = if into { nc / nt } else { nt / nc };
          let ddn = r.d.dot(nl);
          let cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
          if cos2t < 0.0 {
            obj.e + f.mult(radiance(&reflection, depth + 1, xi, spheres))
          } else {
            let tdir = (r.d * nnt - n * (if into { 1.0 } else { -1.0 }
                       * (ddn * nnt + cos2t.sqrt()))).norm();
            let ray = Ray { o: x, d: tdir };
            let a = nt - nc;
            let b = nt + nc;
            let r0 = (a * a) / (b * b);
            let c = 1.0 - (if into {-ddn}else{tdir.dot(n)});
            let re = r0 + (1.0 - r0) * c * c * c * c * c;
            let tr = 1.0 - re;
            let p = 0.25 + 0.5 * re;
            let rp = re / p;
            let tp = tr / (1.0 - p);
            let russian = if depth > 1 {
              if rand::random::<f64>() < p {
                radiance(&reflection, depth + 1, xi, spheres) * rp
              } else {
                radiance(&ray, depth + 1, xi, spheres) * tp
              }
            } else {
              radiance(&reflection, depth + 1, xi, spheres) * re
                + radiance(&ray, depth + 1, xi, spheres) * tr
            };
            russian
          }
        }
      };
      obj.e + f.mult(ir)
    },
  };
  est_radiance
}

fn main() {
  let spheres = [
    Sphere { rad: 1e5, p: V(1e5+1.0,40.8,81.6), e: V(0.0,0.0,0.0), c: V(0.75,0.25,0.25), refl: Reflection::DIFF }, //Left
    Sphere { rad: 1e5,  p: V(-1e5+99.0,40.8,81.6),e: V(0.0,0.0,0.0), c: V(0.25,0.25,0.75), refl: Reflection::DIFF},//Rght
    Sphere { rad: 1e5,  p: V(50.0, 40.8, 1e5),    e: V(0.0,0.0,0.0), c: V(0.75,0.75,0.75), refl: Reflection::DIFF},//Back
    Sphere { rad: 1e5,  p: V(50.0, 40.8,-1e5+170.0),e: V(0.0,0.0,0.0),c:V(0.0,0.0,0.0),   refl: Reflection::DIFF},//Frnt
    Sphere { rad: 1e5,  p: V(50.0,  1e5, 81.6),   e: V(0.0,0.0,0.0), c:V(0.75,0.75,0.75), refl: Reflection::DIFF},//Botm
    Sphere { rad: 1e5,  p: V(50.0, -1e5+81.6,81.6),e:V(0.0,0.0,0.0),c:V(0.75,0.75,0.75), refl: Reflection::DIFF},//Top
    Sphere { rad: 16.5, p: V(27.0, 16.5,47.0),      e: V(0.0,0.0,0.0),c:V(1.0,1.0,1.0)*0.999,  refl: Reflection::SPEC},//Mirr
    Sphere { rad: 16.5, p: V(73.0, 16.5,78.0),       e:V(0.0,0.0,0.0),c:V(1.0,1.0,1.0)*0.999,  refl: Reflection::REFR},//Glas
    Sphere { rad: 600.0,  p: V(50.0, 681.6-0.27, 81.6),e:V(12.0,12.0,12.0),  c: V(0.0,0.0,0.0),  refl: Reflection::DIFF} //Lite
  ];

  let width = 1024;
  let height = 768;
  let samples = match std::env::args().skip(1).next() {
    Some(s) => u32::from_str(&s).unwrap_or(4) / 4,
    None => 1,
  };

  let cam = Ray { o: V(50.0, 52.0, 295.6), d: V(0.0, -0.042612, -1.0).norm() };
  let cx = V(width as f64 * 0.5135 / height as f64, 0.0, 0.0);
  let cy = (cx % cam.d).norm() * 0.5135;
  let mut pixels = vec![V(0.0, 0.0, 0.0); width * height];

  let calc_pixel_value = |x, y| {
    let mut c = V(0.0, 0.0, 0.0);
    for sy in 0..2 {
      for sx in 0..2 {
        let mut r = V(0.0, 0.0, 0.0);
        for _ in 0..samples {
          let r1 = 2.0 * rand::random::<f64>();
          let dx = if r1 < 1.0 { r1.sqrt() - 1.0 } else { 1.0 - (2.0 - r1).sqrt() };
          let r2 = 2.0 * rand::random::<f64>();
          let dy = if r2 < 1.0 { r2.sqrt() - 1.0 } else { 1.0 - (2.0 - r2).sqrt() };
          let d = cx * (((sx as f64 + 0.5 + dx) / 2.0 + x as f64) / width as f64 - 0.5)
                + cy * (((sy as f64 + 0.5 + dy) / 2.0 + y as f64) / height as f64 - 0.5)
                + cam.d;
          let ray2 = Ray { o: cam.o + d * 130.0, d: d.norm() };
          r = r + radiance(&ray2, 0, (0,0,0), &spheres) * (1.0 / (samples * 50) as f64);
          c = c + V(clamp(r.0), clamp(r.1), clamp(r.2)) * 0.25;
        }
      }
    }
    c
  };

  {
    let bands: Vec<(usize, &mut[V])> = pixels.chunks_mut(width).enumerate().collect();
    bands.into_par_iter().for_each(|(y, band)| {
      print!("\rRendering ({} spp) {:10.7}%", samples * 4,
            100.0 * (y as f64 / (height - 1) as f64));
      std::io::stdout().flush().unwrap();

      (0..band.len()).zip(band).for_each(|(x, buf)| {
        let _i = (height - y - 1) * width + x;
        *buf = calc_pixel_value(x, height - y);
      });
    });
  }

  println!("\nRendering completed. Saving into a file...");

  let filename = format!("images/rs.{}.ppm", samples);
  let mut f = BufWriter::new(fs::File::create(filename).unwrap());
  f.write(format!("P3\n{} {}\n{}\n", width, height, 255).as_bytes())
   .expect("Failed to write PPM header");
  for color in pixels {
    f.write(format!("{} {} {} ", to_int(color.0), to_int(color.1), to_int(color.2)).as_bytes())
     .expect("Failed to write out PPM data.");
  }

}
