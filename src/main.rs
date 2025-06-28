use core::f32;
use std::{
    f32::INFINITY,
    ops::{Add, Mul},
    sync::{atomic::AtomicI32, Arc, Mutex},
};

use image::{
    imageops::rotate90, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Rgb,
    RgbImage,
};
use nalgebra_glm::*;
use rayon::{prelude::*, vec};
use rendering::{Camera, FaceIndex, Object3D, Renderer};
mod rendering;

type Triangle = [(Vec3, Vec3); 3];

#[derive(Clone)]
struct MeshObject {
    name: String,
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    faces: Vec<[FaceIndex; 3]>,
    transform: Mat4,
}

impl Object3D for MeshObject {
    fn get_object_id(&self) -> &str {
        &self.name
    }
    fn get_face(&self, i: usize) -> [FaceIndex; 3] {
        self.faces[i]
    }

    fn get_vertex(&self, i: usize) -> Vec3 {
        self.vertices[i]
    }

    fn get_normal(&self, i: usize) -> Vec3 {
        self.normals[i]
    }

    fn get_model_transform(&self) -> &Mat4 {
        &self.transform
    }

    fn get_vertex_count(&self) -> usize {
        self.vertices.len()
    }

    fn get_face_count(&self) -> usize {
        self.faces.len()
    }
}

fn load_obj(path: &str) -> Vec<MeshObject> {
    let mut objs = vec![];
    let mut vertex_idx_offset = 0;
    let mut normal_idx_offset = 0;
    let mut obj = MeshObject {
        name: "monke".to_owned(),
        vertices: vec![],
        normals: vec![],
        faces: vec![],
        transform: identity(),
    };

    let s = std::fs::read_to_string(path).unwrap();

    for line in s.lines() {
        let words: Vec<_> = line.split_whitespace().collect();

        match words[0] {
            "o" => {
                if obj.faces.len() > 0 {
                    println!("Pushing object {}", obj.name);
                    vertex_idx_offset += obj.vertices.len();
                    normal_idx_offset += obj.normals.len();
                    objs.push(obj)
                }

                let object_name = words[1];
                obj = MeshObject {
                    name: object_name.to_owned(),
                    vertices: vec![],
                    normals: vec![],
                    faces: vec![],
                    transform: identity(),
                }
            }
            "v" => {
                let x = words[1].parse::<f32>().unwrap();
                let y = words[2].parse::<f32>().unwrap();
                let z = words[3].parse::<f32>().unwrap();

                obj.vertices.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let x = words[1].parse::<f32>().unwrap();
                let y = words[2].parse::<f32>().unwrap();
                let z = words[3].parse::<f32>().unwrap();

                obj.normals.push(normalize(&Vec3::new(x, y, z)));
            }
            "f" => {
                let [v1, n1] = words[1]
                    .split("//")
                    .map(|n| n.parse::<usize>().unwrap())
                    .collect::<Vec<usize>>()[..]
                else {
                    unreachable!()
                };
                let [v2, n2] = words[2]
                    .split("//")
                    .map(|n| n.parse::<usize>().unwrap())
                    .collect::<Vec<usize>>()[..]
                else {
                    unreachable!()
                };
                let [v3, n3] = words[3]
                    .split("//")
                    .map(|n| n.parse::<usize>().unwrap())
                    .collect::<Vec<usize>>()[..]
                else {
                    unreachable!()
                };

                obj.faces.push([
                    FaceIndex::new(v1 - 1 - vertex_idx_offset, n1 - 1 - normal_idx_offset),
                    FaceIndex::new(v2 - 1 - vertex_idx_offset, n2 - 1 - normal_idx_offset),
                    FaceIndex::new(v3 - 1 - vertex_idx_offset, n3 - 1 - normal_idx_offset),
                ]);
            }
            _ => eprintln!("Unknown element"),
        }
    }

    if obj.faces.len() > 0 {
        println!("Pushing object {}", obj.name);
        objs.push(obj)
    }

    return objs;
}

/*
*  screen space is (0, 0) at center, 1 at each corner. top right is (1, 1)
*  image coords are (0, 0) at top left, (w, h) at bottom right)
*/

fn s_to_i(s: Vec2, w: u32, h: u32) -> (u32, u32) {
    let i = Vec2::new((s.x + 1.0) * 0.5 * w as f32, (1.0 - s.y) * 0.5 * h as f32);

    (i.x.round() as u32, i.y.round() as u32)
}

fn i_to_s(s: (u32, u32), w: u32, h: u32) -> Vec2 {
    vec2(
        (s.0 as f32 / w as f32) * 2.0 - 1.0,
        (1.0 - (s.1 as f32 / h as f32)) * 2.0 - 1.0,
    )
}

fn rotz(v: Vec3, a: f32) -> Vec3 {
    mat3(a.cos(), -a.sin(), 0., a.sin(), a.cos(), 0., 0., 0., 1.) * v
}

fn m_rotz(a: f32) -> Mat3 {
    mat3(a.cos(), -a.sin(), 0., a.sin(), a.cos(), 0., 0., 0., 1.)
    // identity()
}

fn m_roty(a: f32) -> Mat3 {
    mat3(a.cos(), 0., a.sin(), 0., 1.0, 0., -a.sin(), 0., a.cos())
}

use std::f32::consts::*;

fn edge(v0: Vec2, v1: Vec2, p: Vec2) -> f32 {
    (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x)
}

fn pt_inside_tri(v0: Vec2, v1: Vec2, v2: Vec2, p: Vec2) -> bool {
    let ccw = edge(v0, v1, p) < 0. && edge(v1, v2, p) < 0. && edge(v2, v0, p) < 0.;
    let cw = edge(v0, v1, p) > 0. && edge(v1, v2, p) > 0. && edge(v2, v0, p) > 0.;

    ccw || cw
}

fn main() {
    let objs = load_obj("./scene.obj");
    let (width, height) = (512, 512);

    let fps = 24;
    let duration = 5;
    let dt = 1.0 / fps as f32;

    let bloom_threshold = 0.32;
    let bloom_strength = 1.0;
    let bloom_radius = 18i32;

    fn gaussian_kernel_2d(radius: usize, sigma: f32) -> Vec<Vec<f32>> {
        let size = 2 * radius + 1;
        let mut kernel = vec![vec![0.0; size]; size];
        let two_sigma_sq = 2.0 * sigma * sigma;
        let mut sum = 0.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as isize - radius as isize;
                let dy = y as isize - radius as isize;
                let value = (-((dx * dx + dy * dy) as f32) / two_sigma_sq).exp();
                kernel[y][x] = value;
                sum += value;
            }
        }

        for row in &mut kernel {
            for val in row {
                *val /= sum;
            }
        }

        kernel
    }

    let blur_kernel = gaussian_kernel_2d(bloom_radius as usize, 32.0);

    let frame_complete_count = AtomicI32::new(0);

    (0..fps * duration).for_each(|frame| {
        let camera_pos = vec3(0f32, 1f32, 4f32);
        let camera_transform = m_roty(frame as f32 * 2.0 * PI / (fps as f32 * duration as f32));

        let camera_pos = camera_transform * camera_pos;

        let camera = Camera::new(
            vec4(camera_pos.x, camera_pos.y, camera_pos.z, 1.0f32),
            vec4(0f32, 0f32, 0f32, 1f32),
            vec3(0f32, 1f32, 0f32),
            80f32,
            0.1,
            10.0,
        );

        let mut renderer = Renderer::new(width as i32, height as i32);

        for obj in objs.clone() {
            renderer.add_object(obj);
        }

        renderer.render(&camera);

        // renderer
        // .save(&format!(
        //     "./out/frame-{}.png",
        //     frame_complete_count.load(std::sync::atomic::Ordering::SeqCst)
        // ))
        // .unwrap();

        let mut bloom_buff = RgbImage::new(width, height);

        for (x, y, pixel) in renderer.render_buff.enumerate_pixels() {
            let brightness = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0 / 255.0;
            if brightness > bloom_threshold {
                let bloom_color = Rgb([
                    (pixel[0] as f32 * bloom_strength) as u8,
                    (pixel[1] as f32 * bloom_strength) as u8,
                    (pixel[2] as f32 * bloom_strength) as u8,
                ]);
                bloom_buff.put_pixel(x, y, bloom_color);
            } else {
                bloom_buff.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }

        let mut blurred_bloom = RgbImage::new(width, height);

        for i in 0i32..(width as i32 * height as i32) {
            let x = i % width as i32;
            let y = i / width as i32;
            let pix = renderer.render_buff.get_pixel(x as u32, y as u32).0;
            let mut r = pix[0] as f32;
            let mut g = pix[1] as f32;
            let mut b = pix[2] as f32;

            for dx in -bloom_radius..=bloom_radius {
                for dy in -bloom_radius..=bloom_radius {
                    let pix_x = x + dx;
                    let pix_y = y + dy;
                    if pix_x < 0 || pix_x >= (width as i32) || pix_y < 0 || pix_y >= (height as i32)
                    {
                        continue;
                    }

                    let blur_x = dx + bloom_radius;
                    let blur_y = dy + bloom_radius;

                    let kernel_val = blur_kernel[blur_x as usize][blur_y as usize];

                    r += bloom_buff.get_pixel(pix_x as u32, pix_y as u32).0[0] as f32 * kernel_val;
                    g += bloom_buff.get_pixel(pix_x as u32, pix_y as u32).0[1] as f32 * kernel_val;
                    b += bloom_buff.get_pixel(pix_x as u32, pix_y as u32).0[2] as f32 * kernel_val;
                }
            }
            blurred_bloom.put_pixel(x as u32, y as u32, Rgb([r as u8, g as u8, b as u8]));
        }

        blurred_bloom
            .save(format!("./out/ouput-{frame}.png"))
            .unwrap();

        frame_complete_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        println!(
            "Completed {}/{} frames",
            frame_complete_count.load(std::sync::atomic::Ordering::SeqCst),
            fps * duration,
        )

        // img.save(format!("./out/output-{frame}.png")).unwrap();
    });
}
