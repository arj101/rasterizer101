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
use rayon::prelude::*;
mod rendering;

type Triangle = [(Vec3, Vec3); 3];

fn load_obj(path: &str) -> Vec<Triangle> {
    let s = std::fs::read_to_string(path).unwrap();

    let mut vertices: Vec<Vec3> = vec![];
    let mut normals: Vec<Vec3> = vec![];

    let mut faces = vec![];

    for line in s.lines() {
        let words: Vec<_> = line.split_whitespace().collect();

        match words[0] {
            "v" => {
                let x = words[1].parse::<f32>().unwrap();
                let y = words[2].parse::<f32>().unwrap();
                let z = words[3].parse::<f32>().unwrap();

                vertices.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let x = words[1].parse::<f32>().unwrap();
                let y = words[2].parse::<f32>().unwrap();
                let z = words[3].parse::<f32>().unwrap();

                normals.push(normalize(&Vec3::new(x, y, z)));
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

                faces.push([
                    (vertices[v1 - 1], normals[n1 - 1]),
                    (vertices[v2 - 1], normals[n2 - 1]),
                    (vertices[v3 - 1], normals[n3 - 1]),
                ]);
            }
            _ => eprintln!("Unknown element"),
        }
    }

    return faces;
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
    let faces = load_obj("./monke.obj");
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

    (0..fps * duration).into_par_iter().for_each(|frame| {
        let mut img = RgbImage::new(width, height);

        let mut depth_buff: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(width, height);

        depth_buff.fill(INFINITY);

        let shade = |view_transform: Mat4, v: Vec4, n: Vec4| -> Vec4 {
            let mut g = 0.;
            let mut color: Vec4 = vec4(0., 0., 0., 0.0);
            // g *= f32::abs(dot(&n, &vec3(0., 0., -1.)));

            let l1d = view_transform * vec4(-1.1, -1.2, 0.0, 0.0);
            let l1p = view_transform * vec4(1.0, 0.0, 1.0, 1.0);

            let mut l1i = dot(&normalize(&n), &normalize(&-l1d));
            l1i *= 1. / distance(&v, &l1p).powi(2);
            color += vec4(1.0, 0.3, 0.5, 0.0).scale(l1i);

            let l1d = view_transform * vec4(2.1, 1.5, -1.0, 0.0);
            let l1p = view_transform * vec4(-1.0, -1.0, 1.0, 1.0);

            let mut l1i = dot(&normalize(&n), &normalize(&-l1d));
            l1i *= 1.2 / distance(&v, &l1p).powi(2);
            color += vec4(0.2, 0.1, 1.2, 0.0).scale(l1i);

            let l1d = view_transform * vec4(0.0, 1.5, -1.0, 0.0);
            let l1p = view_transform * vec4(-0.0, -1.0, 1.0, 1.0);

            let mut l1i = dot(&normalize(&n), &normalize(&-l1d));
            l1i *= 1.2 / distance(&v, &l1p).powi(2);
            color += vec4(0.0, 1.0, 0.0, 0.0).scale(l1i);

            // vec3(g, g, g )
            color

            // vec4(1.0, 1.0, 1.0, 0.0)
        };

        let camera_transform = m_roty(frame as f32 * 2.0 * PI / (fps as f32 * duration as f32));

        let camera_pos = vec3(0.0f32, 0.0, 2.0);
        let camera_pos = camera_transform * camera_pos;

        let look_direction = vec3(0.0f32, 0.0, 0.0);

        let model_transform = identity();
        // let model_transform = rotate_y(
        //     &model_transform,
        //     frame as f32 * 2.0 * PI / (fps as f32 * duration as f32),
        // );

        // let model_transform = m_roty(0.0);
        let view_transform = look_at(&camera_pos, &look_direction, &vec3(0.0f32, 1.0, 0.0));

        let projection_transform =
            perspective_fov(60f32 * f32::consts::PI / 180f32, 1f32, 1f32, 0.1f32, 10f32);

        let model_view = view_transform * model_transform;

        let normal_transform = model_view.try_inverse().unwrap().transpose();

        let inv_view_transform = (projection_transform).try_inverse().unwrap(); //takes NDC to View space, during interpolation to give vertex coords to shader

        for (face_num, face) in faces.iter().enumerate() {
            // if face_num > frame {
            //     continue;
            // }
            let mut v1 = m_rotz(-FRAC_PI_2) * face[0].0.scale(0.6);
            let mut v2 = m_rotz(-FRAC_PI_2) * face[1].0.scale(0.6);
            let mut v3 = m_rotz(-FRAC_PI_2) * face[2].0.scale(0.6);

            // let model_transform = m_roty(frame as f32 * 2.0 * PI / (fps as f32 * duration as f32));

            // let mvp = projection_transform

            let mut v1 = vec4(v1.x, v1.y, v1.z, 1.0);
            let mut v2 = vec4(v2.x, v2.y, v2.z, 1.0);
            let mut v3 = vec4(v3.x, v3.y, v3.z, 1.0);
            v1 = model_view * v1;
            v2 = model_view * v2;
            v3 = model_view * v3;

            let view_v1 = v1;
            let view_v2 = v2;
            let view_v3 = v3;

            let v1 = projection_transform * v1;
            let v2 = projection_transform * v2;
            let v3 = projection_transform * v3;

            let n1 = m_rotz(-FRAC_PI_2) * face[0].1;
            let n2 = m_rotz(-FRAC_PI_2) * face[1].1;
            let n3 = m_rotz(-FRAC_PI_2) * face[2].1;

            let n1 = vec4(n1.x, n1.y, n1.z, 0.0);
            let n2 = vec4(n2.x, n2.y, n2.z, 0.0);
            let n3 = vec4(n3.x, n3.y, n3.z, 0.0);

            let n1 = normal_transform * n1;
            let n2 = normal_transform * n2;
            let n3 = normal_transform * n3;

            let cam_z = 2.5;
            // v1.z -= cam_z;
            // v2.z -= cam_z;
            // v3.z -= cam_z;

            // v1.x *= near / v1.z;
            // v1.y *= near / v1.z;

            // v2.x *= near / v2.z;
            // v2.y *= near / v2.z;

            // v3.x *= near / v3.z;
            // v3.y *= near / v3.z;

            let v1 = v1.xyz() / v1.w;
            let v2 = v2.xyz() / v2.w;
            let v3 = v3.xyz() / v3.w;

            let (mut x_min, mut y_min) = (INFINITY, INFINITY);
            let (mut x_max, mut y_max) = (-INFINITY, -INFINITY);

            if v1.x < x_min {
                x_min = v1.x;
            }
            if v1.y < y_min {
                y_min = v1.y;
            }
            if v2.x < x_min {
                x_min = v2.x;
            }
            if v2.y < y_min {
                y_min = v2.y
            }
            if v3.x < x_min {
                x_min = v3.x;
            }
            if v3.y < y_min {
                y_min = v3.y;
            }

            if v1.x > x_max {
                x_max = v1.x;
            }
            if v2.x > x_max {
                x_max = v2.x;
            }
            if v3.x > x_max {
                x_max = v3.x;
            }
            if v1.y > y_max {
                y_max = v1.y;
            }
            if v2.y > y_max {
                y_max = v2.y;
            }
            if v3.y > y_max {
                y_max = v3.y;
            }

            let imin = s_to_i(vec2(x_min, y_min), width, height);
            let imax = s_to_i(vec2(x_max, y_max), width, height);

            // println!("({}, {}), ({}, {})", imin.0, imin.1, imax.0, imax.1);

            for i in imin.0..=imax.0 {
                for j in imax.1..=imin.1 {
                    let p = i_to_s((i, j), width, height);

                    if i >= width || j >= height {
                        continue;
                    }
                    // println!("{}, {}, {}", p, i, j);
                    //
                    let area = edge(v1.xy(), v2.xy(), v3.xy());

                    if area == 0.0 {
                        continue;
                    }

                    let w0 = edge(v2.xy(), v3.xy(), p) / area;
                    let w1 = edge(v3.xy(), v1.xy(), p) / area;
                    let w2 = edge(v1.xy(), v2.xy(), p) / area;

                    let z = 1.0 / (w0 / v1.z + w1 / v2.z + w2 / v3.z);

                    let x = w0 * v1.x + w1 * v2.x + w2 * v3.x;
                    let y = w0 * v1.y + w1 * v2.y + w2 * v3.y;

                    let nx = z * (n1.x * w0 / z + n2.x * w1 / z + n3.x * w2 / z);
                    let ny = z * (n1.y * w0 / z + n2.y * w1 / z + n3.y * w2 / z);
                    let nz = z * (n1.z * w0 / z + n2.z * w1 / z + n3.z * w2 / z);
                    //
                    // let n = if area > 0.0 { cross(&(v2 - v1), &(v3-v2)) } else { -cross(&(v3-v2), &(v2-v1)) };

                    let n = vec4(nx, ny, nz, 0.0);

                    let ndc_point = vec4(x, y, z, 1.0);
                    let view_point = inv_view_transform * ndc_point;
                    let world_point = view_point / view_point.w;

                    // println!("{}", vz);

                    if pt_inside_tri(v1.xy(), v2.xy(), v3.xy(), p)
                        && depth_buff.get_pixel(i, j).0[0] > z
                    {
                        let c = shade(view_transform, world_point, n);
                        img.put_pixel(
                            i,
                            j,
                            Rgb([
                                (c.x.clamp(0., 1.0) * 255.0).round() as u8,
                                (c.y.clamp(0., 1.0) * 255.0).round() as u8,
                                (c.z.clamp(0., 1.0) * 255.0).round() as u8,
                            ]),
                        );
                        depth_buff.put_pixel(i, j, Luma([z]));

                        // if (c.x >= 1.0 && c.y >= 1.0 && c.z >= 1.0) {
                        //     println!("{}", vec3(nx, ny, nz));
                        // }
                    }
                }
            }

            // let (i, j) = s_to_i(v1.xy(), width, height);
            // img.put_pixel(i, j, Rgb([255, 255, 200]));
            //
            //
            // let (i, j) = s_to_i(v2.xy(), width, height);
            // img.put_pixel(i, j, Rgb([255, 255, 200]));
            //
            // let (i, j) = s_to_i(v3.xy(), width, height);
            // img.put_pixel(i, j, Rgb([255, 255, 200]));
        }

        let mut bloom_buff = RgbImage::new(width, height);

        for (x, y, pixel) in img.enumerate_pixels() {
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
            let pix = img.get_pixel(x as u32, y as u32).0;
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
