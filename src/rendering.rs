use core::f32;
use std::f32::consts::PI;
use std::f32::INFINITY;

use image::{ImageBuffer, ImageResult, Luma, Rgb, Rgb32FImage};
use nalgebra_glm::*;
use nalgebra_glm::{look_at, perspective_fov, vec2, vec4, Mat4, Vec2, Vec3, Vec4};

pub struct Triangle {
    v1: Vec4,
    v2: Vec4,
    v3: Vec4,
}

/*
*  screen space is (0, 0) at center, 1 at each corner. top right is (1, 1)
*  image coords are (0, 0) at top left, (w, h) at bottom right)
*/

fn s_to_i(s: Vec2, w: u32, h: u32) -> (i32, i32) {
    let i = Vec2::new((s.x + 1.0) * 0.5 * w as f32, (1.0 - s.y) * 0.5 * h as f32);

    (i.x.round() as i32, i.y.round() as i32)
}

fn i_to_s(s: (i32, i32), w: i32, h: i32) -> Vec2 {
    vec2(
        (s.0 as f32 / w as f32) * 2.0 - 1.0,
        (1.0 - (s.1 as f32 / h as f32)) * 2.0 - 1.0,
    )
}

fn edge(v0: Vec2, v1: Vec2, p: Vec2) -> f32 {
    (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x)
}

fn pt_inside_tri(v0: Vec2, v1: Vec2, v2: Vec2, p: Vec2) -> bool {
    let ccw = edge(v0, v1, p) < 0. && edge(v1, v2, p) < 0. && edge(v2, v0, p) < 0.;
    let cw = edge(v0, v1, p) > 0. && edge(v1, v2, p) > 0. && edge(v2, v0, p) > 0.;

    ccw || cw
}

#[derive(Copy, Clone)]
pub struct FaceIndex {
    v: usize,
    n: usize,
}

impl FaceIndex {
    pub fn new(v: usize, n: usize) -> Self {
        FaceIndex { v, n }
    }
}
pub trait Object3D {
    fn get_object_id(&self) -> &str;

    fn get_vertex_count(&self) -> usize;
    fn get_vertex(&self, i: usize) -> Vec3;
    fn get_normal(&self, i: usize) -> Vec3;
    fn get_color(&self, i: usize) -> Vec3 {
        _ = i;
        todo!()
    }

    fn get_face_count(&self) -> usize;
    fn get_face(&self, i: usize) -> [FaceIndex; 3];

    fn get_model_transform(&self) -> &Mat4;
}

pub struct Camera {
    position: Vec4,
    center: Vec4,
    up: Vec3,
    fov: f32,
    near: f32,
    far: f32,
    view_mat: Mat4,
    view_proj_mat: Mat4,
    proj_mat: Mat4,
}

impl Camera {
    pub fn new(position: Vec4, center: Vec4, up: Vec3, fov: f32, near: f32, far: f32) -> Self {
        let fov = fov * f32::consts::PI / 180f32;
        Self {
            position,
            center,
            up,
            fov,
            near,
            far,
            view_mat: look_at(&position.xyz(), &center.xyz(), &up),
            view_proj_mat: perspective_fov(fov, 2f32, 2f32, near, far)
                * look_at(&position.xyz(), &center.xyz(), &up),
            proj_mat: perspective_fov(fov, 2f32, 2f32, near, far),
        }
    }

    pub fn rebuild_mats(&mut self) {
        self.view_proj_mat = perspective_fov(self.fov, 1f32, 1f32, self.near, self.far)
            * look_at(&self.position.xyz(), &self.center.xyz(), &self.up);
        self.view_mat = look_at(&self.position.xyz(), &self.center.xyz(), &self.up);
        self.proj_mat = perspective_fov(self.fov, 1f32, 1f32, self.near, self.far);
    }

    pub fn move_to(&mut self, pos: Vec4) {
        self.position = pos;
    }

    pub fn look_at(&mut self, pos: Vec4) {
        self.center = pos;
    }

    pub fn transform_position(&mut self, transform: Mat4) {
        self.position = transform * self.position;
    }

    pub fn transform_center(&mut self, transform: Mat4) {
        self.center = transform * self.center;
    }

    #[inline]
    pub fn build_mvp_transform(&self, model_matrix: &Mat4) -> Mat4 {
        self.view_proj_mat * model_matrix
    }

    #[inline]
    pub fn build_normal_transform(&self, model_matrix: &Mat4) -> Mat4 {
        return (self.view_mat * model_matrix)
            .try_inverse()
            .unwrap()
            .transpose();
    }

    #[inline]
    pub fn build_inv_view_transform(&self) -> Mat4 {
        return self.proj_mat.try_inverse().unwrap();
    }

    #[inline]
    pub fn project_point(mvp: &Mat4, pt: &Vec4) -> Vec4 {
        let pt = mvp * pt;
        return pt;
        // return pt.xyz() / pt.w;
    }

    #[inline]
    pub fn transform_normal(normal_transform: &Mat4, normal: &Vec4) -> Vec4 {
        normal_transform * normal
    }
}

pub struct Renderer<'a> {
    pub render_buff: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub depth_buff: ImageBuffer<Luma<f32>, Vec<f32>>,
    width: i32,
    height: i32,

    objects: Vec<Box<dyn Object3D + 'a>>,
}

fn shade(view_transform: Mat4, v: Vec4, n: Vec4) -> Vec4 {
    let mut g = 0.;
    let mut color: Vec4 = vec4(0., 0., 0., 0.0);
    // g *= f32::abs(dot(&n, &vec3(0., 0., -1.)));

    let l1d = view_transform * vec4(-1.1, -1.2, 0.0, 0.0);
    let l1p = view_transform * vec4(1.0, 0.0, 1.0, 1.0);

    let mut l1i = dot(&normalize(&n), &normalize(&-l1d));
    l1i *= 4. / distance(&v, &l1p).powi(2);
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
}

#[derive(Clone)]
struct ClipSpaceObject {
    name: String,
    vertices: Vec<Vec4>,
    normals: Vec<Vec3>,
    faces: Vec<[FaceIndex; 3]>,
    transform: Mat4,
}

impl ClipSpaceObject {
    fn get_object_id(&self) -> &str {
        &self.name
    }
    fn get_face(&self, i: usize) -> [FaceIndex; 3] {
        self.faces[i]
    }

    fn get_vertex(&self, i: usize) -> Vec4 {
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

impl<'a> ClipSpaceObject {
    fn from_object3d<T: Object3D + 'a + ?Sized>(obj: &Box<T>, camera: &Camera) -> Self {
        let mut clipped = ClipSpaceObject {
            name: obj.get_object_id().to_owned(),
            vertices: vec![],
            normals: vec![],
            faces: vec![],
            transform: *obj.get_model_transform(),
        };

        let mvp = camera.build_mvp_transform(obj.get_model_transform());
        let normal_transform = camera.build_normal_transform(obj.get_model_transform());
        for i in 0..obj.get_face_count() {
            let face = obj.get_face(i);
            let v1 = obj.get_vertex(face[0].v);
            let v2 = obj.get_vertex(face[1].v);
            let v3 = obj.get_vertex(face[2].v);

            let n1 = obj.get_normal(face[0].n);
            let n2 = obj.get_normal(face[1].n);
            let n3 = obj.get_normal(face[2].n);

            let v1_clip = Camera::project_point(&mvp, &vec4(v1.x, v1.y, v1.z, 1f32));
            let v2_clip = Camera::project_point(&mvp, &vec4(v2.x, v2.y, v2.z, 1f32));
            let v3_clip = Camera::project_point(&mvp, &vec4(v3.x, v3.y, v3.z, 1f32));

            let v1 = v1_clip;
            let v2 = v2_clip;
            let v3 = v3_clip;

            let n1 =
                Camera::transform_normal(&normal_transform, &vec4(n1.x, n1.y, n1.z, 0f32)).xyz();
            let n2 =
                Camera::transform_normal(&normal_transform, &vec4(n2.x, n2.y, n2.z, 0f32)).xyz();
            let n3 =
                Camera::transform_normal(&normal_transform, &vec4(n3.x, n3.y, n3.z, 0f32)).xyz();

            let mut vis_count = 0;

            if v1_clip.z > camera.near {
                vis_count += 1;
            }
            if v2_clip.z > camera.near {
                vis_count += 1;
            }
            if v3_clip.z > camera.near {
                vis_count += 1;
            }

            //check if any point is behind camera z. if all are behind camera z, discard
            if vis_count <= 0 {
                continue;
            }

            let clip_n = vec3(0., 0., 1.);
            let d = camera.near;
            if vis_count >= 3 {
                clipped.vertices.push(v1);
                clipped.vertices.push(v2);
                clipped.vertices.push(v3);
                clipped.normals.push(n1);
                clipped.normals.push(n2);
                clipped.normals.push(n3);

                clipped.faces.push([
                    FaceIndex::new(clipped.vertices.len() - 3, clipped.normals.len() - 3),
                    FaceIndex::new(clipped.vertices.len() - 2, clipped.normals.len() - 2),
                    FaceIndex::new(clipped.vertices.len() - 1, clipped.normals.len() - 1),
                ]);
            }

            if vis_count == 1 {
                //only one point is visible. need to divide two edges of the triangle

                let mut clip =
                    |vis_pt: Vec4, vis_norm: Vec3, a: Vec4, a_norm: Vec3, b: Vec4, b_norm: Vec3| {
                        let t1 = (-dot(&clip_n, &vis_pt.xyz()) - d)
                            / (dot(&clip_n, &(a.xyz() - vis_pt.xyz())));
                        let t2 = (-dot(&clip_n, &vis_pt.xyz()) - d)
                            / (dot(&clip_n, &(b.xyz() - vis_pt.xyz())));

                        /*
                         *       vis_pt
                         *          / \
                         *==clip== /===\=======
                         *     t1 /     \ t2
                         *       a-------b
                         *
                         */

                        let n2 = vis_norm + (a_norm - vis_norm) * t1;
                        let n3 = vis_norm + (b_norm - vis_norm) * t2;
                        let n1 = vis_norm;

                        let v1 = vis_pt;
                        let v2 = vis_pt + (a - vis_pt) * t1;
                        let v3 = vis_pt + (b - vis_pt) * t2;

                        clipped.vertices.push(v1);
                        clipped.vertices.push(v2);
                        clipped.vertices.push(v3);
                        clipped.normals.push(n1);
                        clipped.normals.push(n2);
                        clipped.normals.push(n3);

                        clipped.faces.push([
                            FaceIndex::new(clipped.vertices.len() - 3, clipped.normals.len() - 3),
                            FaceIndex::new(clipped.vertices.len() - 2, clipped.normals.len() - 2),
                            FaceIndex::new(clipped.vertices.len() - 1, clipped.normals.len() - 1),
                        ]);
                    };

                if v1.z > camera.near {
                    clip(v1, n1, v2, n2, v3, n3);
                } else if v2.z > camera.near {
                    clip(v2, n2, v1, n1, v3, n3);
                } else if v3.z > camera.near {
                    clip(v3, n3, v1, n1, v2, n2);
                }
            }

            if vis_count == 2 {
                let mut clip = |invis_pt: Vec4,
                                invis_norm: Vec3,
                                a: Vec4,
                                a_norm: Vec3,
                                b: Vec4,
                                b_norm: Vec3| {
                    /*
                     *       a-------b
                     *    t1  \     / t2
                     *==clip== \===/=======
                     *          \ /
                     *         invis_pt
                     *
                     */

                    let t1 = (-dot(&clip_n, &invis_pt.xyz()) - d)
                        / (dot(&clip_n, &(a.xyz() - invis_pt.xyz())));
                    let t2 = (-dot(&clip_n, &invis_pt.xyz()) - d)
                        / (dot(&clip_n, &(b.xyz() - invis_pt.xyz())));

                    let v_a = invis_pt + (a - invis_pt) * t1;
                    let v_b = invis_pt + (b - invis_pt) * t2;

                    let n_a = invis_norm + (a_norm - invis_norm) * t1;
                    let n_b = invis_norm + (b_norm - invis_norm) * t2;

                    clipped.vertices.push(a);
                    clipped.vertices.push(b);
                    clipped.vertices.push(v_a);
                    clipped.normals.push(a_norm);
                    clipped.normals.push(b_norm);
                    clipped.normals.push(n_a);

                    clipped.faces.push([
                        FaceIndex::new(clipped.vertices.len() - 3, clipped.normals.len() - 3),
                        FaceIndex::new(clipped.vertices.len() - 2, clipped.normals.len() - 2),
                        FaceIndex::new(clipped.vertices.len() - 1, clipped.normals.len() - 1),
                    ]);

                    clipped.vertices.push(b);
                    clipped.vertices.push(v_b);
                    clipped.vertices.push(v_a);
                    clipped.normals.push(b_norm);
                    clipped.normals.push(n_b);
                    clipped.normals.push(n_a);

                    clipped.faces.push([
                        FaceIndex::new(clipped.vertices.len() - 3, clipped.normals.len() - 3),
                        FaceIndex::new(clipped.vertices.len() - 2, clipped.normals.len() - 2),
                        FaceIndex::new(clipped.vertices.len() - 1, clipped.normals.len() - 1),
                    ]);
                };

                if v1.z <= camera.near {
                    clip(v1, n1, v2, n2, v3, n3);
                }
                if v2.z <= camera.near {
                    clip(v2, n2, v1, n1, v3, n3);
                }
                if v3.z <= camera.near {
                    clip(v3, n3, v1, n1, v2, n2);
                }
            }
        }

        return clipped;
    }
}

impl<'a> Renderer<'a> {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            render_buff: ImageBuffer::new(width as u32, height as u32),
            depth_buff: ImageBuffer::new(width as u32, height as u32),
            width,
            height,
            objects: vec![],
        }
    }

    pub fn add_object<T: Object3D + 'a>(&mut self, object: T) {
        self.objects.push(Box::new(object));
    }

    pub fn save(&mut self, path: &str) -> ImageResult<()> {
        return self.render_buff.save(path);
    }

    pub fn render(&mut self, cam: &Camera) {
        self.render_buff.fill(0); //clear buffer
        self.depth_buff.fill(INFINITY);

        let inv_view_transform = cam.build_inv_view_transform();

        for obj in &self.objects {
            // let mvp = cam.build_mvp_transform(obj.get_model_transform());
            // let normal_transform = cam.build_normal_transform(obj.get_model_transform());

            // let obj = (obj.as_ref());
            let clipped_obj = ClipSpaceObject::from_object3d(obj, cam);

            let mut render_face = |v1: Vec4, v2: Vec4, v3: Vec4, n1: Vec3, n2: Vec3, n3: Vec3| {
                let v1_screen = v1.xyz() / v1.w;
                let v2_screen = v2.xyz() / v2.w;
                let v3_screen = v3.xyz() / v3.w;

                let n1_view = n1;
                let n2_view = n2;
                let n3_view = n3;

                let (mut x_min, mut y_min) = (
                    v1_screen.x.min(v2_screen.x.min(v3_screen.x)),
                    v1_screen.y.min(v2_screen.y.min(v3_screen.y)),
                );
                let (mut x_max, mut y_max) = (
                    v1_screen.x.max(v2_screen.x.max(v3_screen.x)),
                    v1_screen.y.max(v2_screen.y.max(v3_screen.y)),
                );

                let imin = s_to_i(vec2(x_min, y_min), self.width as u32, self.height as u32);
                let imax = s_to_i(vec2(x_max, y_max), self.width as u32, self.height as u32);

                for i in imin.0..=imax.0 {
                    for j in imax.1..=imin.1 {
                        let p = i_to_s((i, j), self.width, self.height);

                        if i >= self.width || j >= self.height || i < 0 || j < 0 {
                            continue;
                        }

                        let i = i as u32;
                        let j = j as u32;
                        // println!("{}, {}, {}", p, i, j);
                        //
                        let area = edge(v1_screen.xy(), v2_screen.xy(), v3_screen.xy());

                        if area == 0.0 {
                            continue;
                        }

                        let w0 = edge(v2_screen.xy(), v3_screen.xy(), p) / area;
                        let w1 = edge(v3_screen.xy(), v1_screen.xy(), p) / area;
                        let w2 = edge(v1_screen.xy(), v2_screen.xy(), p) / area;

                        let z = 1.0 / (w0 / v1_screen.z + w1 / v2_screen.z + w2 / v3_screen.z);

                        let x = w0 * v1_screen.x + w1 * v2_screen.x + w2 * v3_screen.x;
                        let y = w0 * v1_screen.y + w1 * v2_screen.y + w2 * v3_screen.y;

                        let nx = z * (n1_view.x * w0 / z + n2_view.x * w1 / z + n3_view.x * w2 / z);
                        let ny = z * (n1_view.y * w0 / z + n2_view.y * w1 / z + n3_view.y * w2 / z);
                        let nz = z * (n1_view.z * w0 / z + n2_view.z * w1 / z + n3_view.z * w2 / z);
                        //
                        // let n = if area > 0.0 { cross(&(v2 - v1), &(v3-v2)) } else { -cross(&(v3-v2), &(v2-v1)) };

                        let n = vec4(nx, ny, nz, 0.0);

                        let ndc_point = vec4(x, y, z, 1.0);
                        let view_point = inv_view_transform * ndc_point;
                        let world_point = view_point / view_point.w;

                        // println!("{}", vz);

                        if pt_inside_tri(v1_screen.xy(), v2_screen.xy(), v3_screen.xy(), p)
                            && self.depth_buff.get_pixel(i, j).0[0] > z
                        {
                            let c = shade(cam.view_mat, world_point, n);
                            self.render_buff.put_pixel(
                                i,
                                j,
                                Rgb([
                                    (c.x.clamp(0., 1.0) * 255.0).round() as u8,
                                    (c.y.clamp(0., 1.0) * 255.0).round() as u8,
                                    (c.z.clamp(0., 1.0) * 255.0).round() as u8,
                                ]),
                            );
                            self.depth_buff.put_pixel(i, j, Luma([z]));

                            // if (c.x >= 1.0 && c.y >= 1.0 && c.z >= 1.0) {
                            //     println!("{}", vec3(nx, ny, nz));
                            // }
                        }
                    }
                }
            };

            for face_idx in 0..clipped_obj.get_face_count() {
                let face = clipped_obj.get_face(face_idx);
                let v1 = clipped_obj.get_vertex(face[0].v);
                let v2 = clipped_obj.get_vertex(face[1].v);
                let v3 = clipped_obj.get_vertex(face[2].v);

                let n1 = clipped_obj.get_normal(face[0].n);
                let n2 = clipped_obj.get_normal(face[1].n);
                let n3 = clipped_obj.get_normal(face[2].n);
                render_face(v1, v2, v3, n1, n2, n3);
            }
        }
    }
}
