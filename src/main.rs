use bevy::{
    hierarchy::ReportHierarchyIssue,
    math::{dvec3, DMat3, DVec3},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
    window::CursorGrabMode,
};
use bevy_enhanced_input::prelude::*;
use big_space::prelude::*;
use controls::*;
use ico_mesh::*;
use rand_pcg::Pcg32;
use std::hash::Hasher;

type P = i64;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            EnhancedInputPlugin,
            // TODO find out why validation is crashing when we despawn entities
            BigSpacePlugin::<P>::new(false),
        ))
        .insert_resource(ReportHierarchyIssue::<InheritedVisibility>::new(false))
        .insert_resource(PlanetHeightMap(Box::new(NoiseHeightMap)))
        .add_input_context::<SpaceshipControls>()
        .add_observer(translation_input)
        .add_observer(rotation_input)
        .add_observer(speed_input)
        .add_observer(capture_cursor_input)
        .add_systems(Startup, setup)
        .add_systems(Update, adjust_speed_text)
        .add_systems(
            Update,
            (mark_for_subdivision, subdivide_faces, recombine_faces).chain(),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    height_map: Res<PlanetHeightMap>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((SpeedText, Text::new("Speed: 0m/s")));
            parent.spawn(Text::new("Press `Tab` to toggle cursor capture"));
        });
    commands.spawn((
        DirectionalLight { ..default() },
        Transform::IDENTITY.looking_to(Dir3::X, Dir3::Y),
    ));
    commands.spawn_big_space(Grid::<P>::new(1e3, 0.0), |grid| {
        grid.spawn_spatial((
            SpaceshipControls,
            Speed(1e7),
            Camera3d::default(),
            Projection::Perspective(PerspectiveProjection {
                near: 1e-8,
                ..default()
            }),
            Transform::from_xyz(0.0, 0.0, 1e7),
            FloatingOrigin,
        ));

        for face in icosphere(6.4e6) {
            let mesh = face.mesh(height_map.0.as_ref());
            let (cell, offset) = grid.grid().translation_to_grid(mesh.offset);
            grid.spawn_spatial((
                Mesh3d(meshes.add(mesh.mesh)),
                MeshMaterial3d::<StandardMaterial>::default(),
                cell,
                Transform::from_translation(offset),
                BoundingSphere(mesh.radius),
                face,
            ));
        }
    });
}

#[derive(Component)]
struct SpeedText;

#[derive(Component)]
struct Speed(f32);

fn adjust_speed_text(
    speed: Option<Single<&Speed, Changed<Speed>>>,
    mut text: Single<&mut Text, With<SpeedText>>,
) {
    let Some(speed) = speed else {
        return;
    };
    ***text = format!("Speed: {}m/s", speed.0);
}

/// Our height will be simplex noise over an icosphere. This means our top-level grid cells are the
/// 20 triangles that make up an icosahedron, projected onto a sphere. Each progressive noise octave
/// uses cells with half the edge length of the previous, meaning cells for the n-th octave are just
/// the triangles of the n-th icosphere subdivision. Each vertex contributes to the final noise
/// function only in its immediate neighborhood, in this case the 5-6 triangles it touches.
///
/// Here's a list of terms so I have something I can reason about in English:
///
/// - **Contribution function**: The total contribution of a single vertex to the noise map. We get
///   the value of the noise function by evaluating each contribution function at the given point
///   and summing them together. In practice, for a given point, only three contribution functions
///   will ever affect the final value, the others will be zero. So we find which three vertices
///   might be affecting our point, evaluate their contribution functions, and sum them. This
///   function is the product of the *gradient function* and the *falloff function*.
/// - **Gradient function**: Each vertex has a random gradient associated with it. This is where the
///   randomness in the noise comes from. The gradient is in "surface coordinates", meaning it's a
///   2-dimensional direction on the compass. In theory, saying "it's a random direction" is good
///   enough for math. In practice, we actually need those coordinates since we're in a computer and
///   not an idealized math world. To compute this, we sample a random 2-vector from the boundary of
///   a circle. Then we build a 3D basis at the vertex where Y points up. Doesn't matter where the
///   other two point as long as it's consistent. Now we have a random 2D gradient from the
///   surface's perspective. To sample the function, we can just take the dot product between this
///   gradient and the offset to the point we're interested in, in the same basis. To figure out
///   that offset, we can literally just take the 3D offset between the points, normalize it, then
///   scale it by the great circle distance between the two points. Now we just transform it into
///   the local space of that basis we built earlier and take the XZ components. Now we can just
///   take the dot product of the gradient and the offset vector. We also have to scale down by the
///   vertex influence distance to clamp the output to [0, 1].
/// - **Falloff function*: This serves a couple purposes. The first is obviously to constrain the
///   influence of the contribution function to just the neighboring faces. The next important one
///   is to ensure that the noise function is smooth. In particular, we're aiming for C2 continuity.
///   This means that the first and second derivatives are continuous. This property makes it easy
///   to extract normals. We'll typically define this as $(1-d^2)^4$ where d (the great circle
///   distance from the sample point to the vertex) is in [0, 1]. This gives us the continuity we're
///   looking for, it satisfies $f(0)=1$ and $f(1)=0$, and it's cheap to compute.
///
/// So C(x)=G(x)F(x).
/// G(X)=dot(g, d(p-v)/(f|p-v|))
/// F(x)=(1-(d/f)^2)^4
struct NoiseHeightMap;

impl HeightMap for NoiseHeightMap {
    fn properties(&self, location: &Location) -> HeightMapProperties {
        // `result` is going to be the contributions to the value and gradient of the noise function
        // from each vertex. We can just sum the value and gradient to figure out our height and
        // normal.
        let mut height = 0.0;
        let mut contributing_vertices = SubFace::top_level(location.face, location.radius);
        let mut amplitude = 20e3; // 20km is the approximate height variation on Earth
        const OCTAVES: usize = 20;
        for octave in 0..OCTAVES {
            height += amplitude
                * contributing_vertices
                    .coords
                    .map(|corner| {
                        let vertex_coord = corner.position;
                        let distance = vertex_coord.angle_between(location.normalized_position)
                            * location.radius;
                        let influence = location.radius / 2.0f64.powi(octave as i32);
                        let basis = arbitrary_but_consistent_basis(vertex_coord);
                        // hash corner.position by converting each f64 -> bits, then hashing the 3-tuple of u64
                        let mut hasher = std::hash::DefaultHasher::default();
                        for corner in corner.position.to_array() {
                            hasher.write_u64(corner.to_bits());
                        }
                        let seed = hasher.finish();
                        let mut rng = Pcg32::new(seed, 17); // TODO
                        let gradient = Circle::new(1.0).sample_boundary(&mut rng).as_dvec2();
                        let offset =
                            basis.transpose() * (location.normalized_position - vertex_coord);
                        let offset = offset.xz().normalize_or_zero() * distance / influence;
                        let gradient_function_result = gradient.dot(offset);
                        let falloff_function_result =
                            (1.0 - (distance / influence).clamp(0.0, 1.0).powi(2)).powi(4);
                        gradient_function_result * falloff_function_result
                    })
                    .into_iter()
                    .sum::<f64>();

            // Make sure amplitude falls off slower than the chunk size, otherwise we'll never see
            // detail. Also make sure it falls off close to the same speed as chunk size, otherwise
            // we get *really* violent noise at small scales. 0.65 seems to work well.
            amplitude *= 0.65;

            // Here's the trick: our new contributing vertices depend on which sub-triangle we're
            // in. We figure out which sub-triangle we're in based on barycentric coordinates: if
            // any coordinate >0.5, we're in the sub-triangle that shares the vertex corresponding
            // to that coordinate. For example, if our coordinate is (0.6, 0.3, 0.1), we're in the
            // sub-triangle sharing the first vertex. If all coordinates are <0.5, we're in the
            // middle sub-triangle.
            //
            // In order to recursively descend through octaves, we need to update the barycentric
            // coordinates every iteration. We can actually transform barycentric coordinates in the
            // parent triangle's coordinates to barycentric coordinates in the coordinates of any
            // arbitrary sub-triangle as long as we know the barycentric coordinates of its corners.
            // We do this the same way we create any world-to-local matrix: inverting the local-to-world
            // matrix. We treat the parent as the "world transform". The standard basis in barycentric
            // coordinates is just the three corners of the triangle. That means our local-to-world
            // matrix is just the corners of the sub-triangle *in the parent's coordinates*. We already
            // have those!!! So we create a matrix `M` where the columns are the corners of the
            // sub-triangle in the parent's coordinates. We can literally just invert this and *BOOM*,
            // now we can multiply it by any point in "world space", or the parent IcoFace's
            // coordinates and we get the point in the sub-triangle's coordinates. We can then use
            // this to figure out which sub-triangle we're in.

            let sub_triangle_basis = DMat3::from_cols_array_2d(&[
                contributing_vertices.coords[0].barycentric_coords,
                contributing_vertices.coords[1].barycentric_coords,
                contributing_vertices.coords[2].barycentric_coords,
            ]);
            let world_to_local = sub_triangle_basis.inverse();
            let local_coords = world_to_local * DVec3::from(location.barycentric_coords);

            let sub_face_index = if local_coords.x > 0.5 {
                0
            } else if local_coords.y > 0.5 {
                1
            } else if local_coords.z > 0.5 {
                2
            } else {
                3
            };
            contributing_vertices = contributing_vertices.subdivide()[sub_face_index];
        }

        HeightMapProperties {
            height,
            // normal: Dir3::new_unchecked(location.normalized_position.as_vec3()),
        }
    }
}

/// For a given position on the planet, we need a consistent, reproducible basis. There's no
/// one convention that'll work for every point. What we'll do is use a different convention
/// depending on whether we're above or below some arbitrary latitude. If we're below,
/// there's no chance our up vector will coincide with global Y, so we use that. if we're
/// above, there's no chance up will coincide with global X or Z, so we use one of those.
/// I'm picking X because it's the first letter of "Xenonion". We can then use the cross
/// product to find the other basis vectors. This will always give us the same basis for any
/// given point.
pub fn arbitrary_but_consistent_basis(coord: DVec3) -> DMat3 {
    // TODO Remove this check with `DDir3`
    let coord = coord.normalize();
    let up = if coord.dot(DVec3::Y).abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = up.cross(coord).normalize();
    let south = east.cross(coord);
    DMat3::from_cols(east, coord, south)
}

#[cfg(test)]
mod tests {
    use crate::{arbitrary_but_consistent_basis, ico_mesh::VERTICES};

    #[test]
    fn test_basis() {
        for vertex in VERTICES {
            let basis = arbitrary_but_consistent_basis(vertex);
            assert!(basis.col(0).dot(basis.col(1)).abs() < 1e-6);
            assert!(basis.col(1).dot(basis.col(2)).abs() < 1e-6);
            assert!(
                basis.determinant().abs() - 1.0 < 1e-6,
                r#"Basis was orthogonal but columns were not unit length!
Vertex: {vertex:?}
Basis: {basis:#?}"#
            );
        }
    }
}

mod controls {
    use super::*;

    #[derive(Component)]
    pub struct SpaceshipControls;

    impl InputContext for SpaceshipControls {
        fn context_instance(_: &World, _: Entity) -> ContextInstance {
            let mut ctx = ContextInstance::default();
            ctx.bind::<Translation>().to((
                KeyCode::KeyD,
                KeyCode::KeyA.with_modifiers(Negate::all()),
                KeyCode::KeyR.with_modifiers(SwizzleAxis::YXZ),
                KeyCode::KeyF.with_modifiers((SwizzleAxis::YXZ, Negate::all())),
                KeyCode::KeyS.with_modifiers(SwizzleAxis::YZX),
                KeyCode::KeyW.with_modifiers((SwizzleAxis::YZX, Negate::all())),
            ));
            ctx.bind::<Rotation>()
                .to((
                    Input::mouse_motion().with_modifiers((
                        SwizzleAxis::YXZ,
                        Negate::all(),
                        Scale::splat(0.1),
                    )),
                    KeyCode::KeyQ.with_modifiers(SwizzleAxis::YZX),
                    KeyCode::KeyE.with_modifiers((SwizzleAxis::YZX, Negate::all())),
                ))
                .with_modifiers(Negate::x()); // x here is rotation about x, not mouse x motion
            ctx.bind::<AdjustSpeed>()
                .to(Input::mouse_wheel().with_modifiers((SwizzleAxis::YXZ, Scale::splat(2.4))));
            ctx.bind::<ToggleCursorCapture>().to(KeyCode::Tab);
            ctx
        }
    }

    #[derive(Debug, InputAction)]
    #[input_action(output = Vec3)]
    pub struct Translation;

    #[derive(Debug, InputAction)]
    #[input_action(output = Vec3)]
    pub struct Rotation;

    #[derive(Debug, InputAction)]
    #[input_action(output = f32)]
    pub struct AdjustSpeed;

    #[derive(Debug, InputAction)]
    #[input_action(output = bool)]
    pub struct ToggleCursorCapture;

    pub fn translation_input(
        input: Trigger<Fired<Translation>>,
        mut player: Query<(&mut Transform, &Speed)>,
        time: Res<Time>,
    ) {
        let (mut transform, &Speed(speed)) = player.get_mut(input.entity()).unwrap();
        let movement = transform.rotation * (input.value * speed * time.delta_secs());
        transform.translation += movement;
    }

    pub fn rotation_input(
        input: Trigger<Fired<Rotation>>,
        mut player: Query<&mut Transform>,
        time: Res<Time>,
    ) {
        let mut player = player.get_mut(input.entity()).unwrap();
        player.rotation *= Quat::from_scaled_axis(input.value * time.delta_secs());
    }

    pub fn speed_input(input: Trigger<Fired<AdjustSpeed>>, mut player: Query<&mut Speed>) {
        let mut speed = player.get_mut(input.entity()).unwrap();
        speed.0 = speed.0 * (input.value * 0.1).exp();
    }

    pub fn capture_cursor_input(
        _: Trigger<Started<ToggleCursorCapture>>,
        mut window: Single<&mut Window>,
    ) {
        let was_captured = window.cursor_options.grab_mode != CursorGrabMode::None;
        window.cursor_options.grab_mode = if was_captured {
            CursorGrabMode::None
        } else {
            CursorGrabMode::Locked
        };
        window.cursor_options.visible = was_captured;
    }
}

mod ico_mesh {
    use bevy::math::FloatOrd;

    use super::*;

    /// The golden friggin ratio.
    pub const PHI: f64 = 1.618033988749894848204586834365638117720309179805762862135448622;

    /// The un-normalized coordinates for the vertices of a regular icosahedron.
    pub const VERTICES: [DVec3; 12] = [
        dvec3(0.0, -1.0, -PHI),
        dvec3(0.0, -1.0, PHI),
        dvec3(0.0, 1.0, -PHI),
        dvec3(0.0, 1.0, PHI),
        dvec3(-1.0, -PHI, 0.0),
        dvec3(-1.0, PHI, 0.0),
        dvec3(1.0, -PHI, 0.0),
        dvec3(1.0, PHI, 0.0),
        dvec3(-PHI, 0.0, -1.0),
        dvec3(-PHI, 0.0, 1.0),
        dvec3(PHI, 0.0, -1.0),
        dvec3(PHI, 0.0, 1.0),
    ];

    pub const FACES: [IcoFace; 20] = [
        IcoFace::new(0, 2, 10),
        IcoFace::new(0, 10, 6),
        IcoFace::new(0, 6, 4),
        IcoFace::new(0, 4, 8),
        IcoFace::new(0, 8, 2),
        IcoFace::new(3, 1, 11),
        IcoFace::new(3, 11, 7),
        IcoFace::new(3, 7, 5),
        IcoFace::new(3, 5, 9),
        IcoFace::new(3, 9, 1),
        IcoFace::new(2, 7, 10),
        IcoFace::new(2, 5, 7),
        IcoFace::new(8, 5, 2),
        IcoFace::new(8, 9, 5),
        IcoFace::new(4, 9, 8),
        IcoFace::new(4, 1, 9),
        IcoFace::new(6, 1, 4),
        IcoFace::new(6, 11, 1),
        IcoFace::new(10, 11, 6),
        IcoFace::new(10, 7, 11),
    ];

    pub fn icosphere(radius: f64) -> [SubFace; 20] {
        FACES.map(|face| SubFace::top_level(face, radius))
    }

    #[derive(Clone, Copy)]
    pub struct FaceCoord {
        /// Normalized position on the icosphere.
        pub position: DVec3,
        pub barycentric_coords: [f64; 3],
    }

    impl FaceCoord {
        pub fn midpoint(self, right: Self) -> Self {
            Self {
                position: self.position.midpoint(right.position).normalize(),
                barycentric_coords: [
                    (self.barycentric_coords[0] + right.barycentric_coords[0]) / 2.0,
                    (self.barycentric_coords[1] + right.barycentric_coords[1]) / 2.0,
                    (self.barycentric_coords[2] + right.barycentric_coords[2]) / 2.0,
                ],
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct IcoFace {
        pub vertices: [usize; 3],
    }

    impl IcoFace {
        pub const fn new(a: usize, b: usize, c: usize) -> Self {
            Self {
                vertices: [a, b, c],
            }
        }
    }

    /// This represents any sub-face of arbitrary order of an icosphere. A 0th-order sub-face of e.g.
    /// face 0 is face 0. Each face has 4 1st order sub-faces, 16 2nd order sub-faces, and so on.
    #[derive(Component, Clone, Copy)]
    pub struct SubFace {
        pub face: IcoFace,
        pub coords: [FaceCoord; 3],
        pub radius: f64,
    }

    impl SubFace {
        pub fn top_level(face: IcoFace, radius: f64) -> Self {
            let vertices = VERTICES.map(|x| x.normalize());
            Self {
                face,
                coords: [
                    FaceCoord {
                        position: vertices[face.vertices[0]],
                        barycentric_coords: [1.0, 0.0, 0.0],
                    },
                    FaceCoord {
                        position: vertices[face.vertices[1]],
                        barycentric_coords: [0.0, 1.0, 0.0],
                    },
                    FaceCoord {
                        position: vertices[face.vertices[2]],
                        barycentric_coords: [0.0, 0.0, 1.0],
                    },
                ],
                radius,
            }
        }

        pub fn subdivide(self) -> [Self; 4] {
            let ab = self.coords[0].midpoint(self.coords[1]);
            let bc = self.coords[1].midpoint(self.coords[2]);
            let ca = self.coords[2].midpoint(self.coords[0]);
            [
                Self {
                    coords: [self.coords[0], ab, ca],
                    ..self
                },
                Self {
                    coords: [self.coords[1], bc, ab],
                    ..self
                },
                Self {
                    coords: [self.coords[2], ca, bc],
                    ..self
                },
                Self {
                    coords: [ab, bc, ca],
                    ..self
                },
            ]
        }

        pub fn mesh(self, height_map: &dyn HeightMap) -> BigMesh {
            struct IndexedFace([usize; 3]);

            impl IndexedFace {
                fn subdivide(self, vertices: &mut Vec<FaceCoord>) -> [Self; 4] {
                    let [a, b, c] = self.0;
                    vertices.push(vertices[a].midpoint(vertices[b]));
                    let ab = vertices.len() - 1;
                    vertices.push(vertices[b].midpoint(vertices[c]));
                    let bc = vertices.len() - 1;
                    vertices.push(vertices[c].midpoint(vertices[a]));
                    let ca = vertices.len() - 1;
                    [
                        Self([a, ab, ca]),
                        Self([b, bc, ab]),
                        Self([c, ca, bc]),
                        Self([ab, bc, ca]),
                    ]
                }
            }

            let mut vertices = self.coords.to_vec();
            let mut faces = vec![IndexedFace([0, 1, 2])];
            const SUBDIVISIONS: usize = 4;
            for _ in 0..SUBDIVISIONS {
                faces = faces
                    .into_iter()
                    .flat_map(|x| x.subdivide(&mut vertices))
                    .collect();
            }
            let faces = faces
                .into_iter()
                .flat_map(|IndexedFace(x)| x.map(|x| x as u32)) // Flatten [[usize; 3]] -> [u32]
                .collect::<Vec<_>>();
            let vertices = vertices
                .into_iter()
                .map(|x| {
                    let props = height_map.properties(&Location {
                        normalized_position: x.position,
                        barycentric_coords: x.barycentric_coords,
                        face: self.face,
                        radius: self.radius,
                    });
                    x.position * (self.radius + props.height)
                })
                .collect::<Vec<_>>();

            let offset = vertices.iter().sum::<DVec3>() / (vertices.len() as f64);
            let uvs = vec![Vec2::ZERO; vertices.len()];
            // let normals = vertices.iter().map(|(_, x)| *x.normal).collect::<Vec<_>>();
            let vertices = vertices
                .iter()
                .map(|x| (x - offset).as_vec3())
                .collect::<Vec<_>>();
            let FloatOrd(radius) = vertices.iter().map(|x| FloatOrd(x.length())).max().unwrap();

            let mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
                .with_inserted_indices(Indices::U32(faces))
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
                // .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
                .with_duplicated_vertices()
                .with_computed_flat_normals();

            BigMesh {
                mesh,
                offset,
                radius,
            }
        }
    }

    #[derive(Clone)]
    pub struct BigMesh {
        pub mesh: Mesh,
        pub offset: DVec3,
        pub radius: f32,
    }

    // If the floating origin gets close to us, we want to add a marker that says we need to be
    // subdivided. But how do we know if we're already subdivided? We create another marker to track
    // that and keep it in sync. That means we need a system to sync the "needs subdivision" marker and
    // one to sync the "is subdivided" marker. We also need a system to subdivide faces with the "needs
    // subdivision" marker and no "is subdivided" marker, and one to remove the face's children when it
    // has the "is subdivided" marker and no "needs subdivision" marker.

    #[derive(Component)]
    pub struct NeedsSubdivision;

    #[derive(Component)]
    pub struct IsSubdivided([Entity; 4]);

    pub fn mark_for_subdivision(
        floating_origin: Single<&GlobalTransform, With<FloatingOrigin>>,
        faces: Query<(Entity, &GlobalTransform, &BoundingSphere)>,
        mut commands: Commands,
    ) {
        let floating_origin = floating_origin.translation();
        for (e, face_transform, &BoundingSphere(radius)) in &faces {
            let needs_subdivision =
                floating_origin.distance(face_transform.translation()) < radius * 2.0;
            if needs_subdivision {
                commands.entity(e).insert(NeedsSubdivision);
            } else {
                commands.entity(e).remove::<NeedsSubdivision>();
            }
        }
    }

    #[derive(Resource)]
    pub struct PlanetHeightMap(pub Box<dyn HeightMap + Send + Sync>);

    pub fn subdivide_faces(
        faces: Query<
            (Entity, &SubFace, &BoundingSphere),
            (With<NeedsSubdivision>, Without<IsSubdivided>),
        >,
        grid_root: Single<(Entity, &Grid<P>), With<BigSpace>>,
        height_map: Res<PlanetHeightMap>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut commands: Commands,
    ) {
        let (grid_root_entity, grid_root) = *grid_root;
        for (e, sub_face, &BoundingSphere(radius)) in &faces {
            // Limit the resolution so we don't subdivide into oblivion
            if radius < 1.0 {
                continue;
            }
            let sub_faces = sub_face.subdivide().map(|face| {
                let mesh = face.mesh(height_map.0.as_ref());
                let (cell, offset) = grid_root.translation_to_grid(mesh.offset);
                commands
                    .spawn((
                        Mesh3d(meshes.add(mesh.mesh)),
                        MeshMaterial3d::<StandardMaterial>::default(),
                        cell,
                        Transform::from_translation(offset),
                        BoundingSphere(mesh.radius),
                        face,
                    ))
                    .id()
            });
            commands
                .entity(e)
                .insert((IsSubdivided(sub_faces), Visibility::Hidden));
            commands.entity(grid_root_entity).add_children(&sub_faces);
        }
    }

    pub fn recombine_faces(
        faces: Query<(Entity, &IsSubdivided), (With<IsSubdivided>, Without<NeedsSubdivision>)>,
        mut commands: Commands,
    ) {
        for (e, &IsSubdivided(sub_faces)) in &faces {
            commands
                .entity(e)
                .remove::<IsSubdivided>()
                // we use `try_insert` here because the face might already have been despawned below in
                // a previous loop iteration
                .try_insert(Visibility::Inherited);
            for sub_face in sub_faces {
                commands.entity(sub_face).despawn();
            }
        }
    }

    #[derive(Component, Clone, Copy)]
    pub struct BoundingSphere(pub f32);

    pub trait HeightMap {
        fn properties(&self, location: &Location) -> HeightMapProperties;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Location {
        pub normalized_position: DVec3,
        pub barycentric_coords: [f64; 3],
        pub face: IcoFace,
        // /// The approximate distance between sample points. This is important to sample maps at the
        // /// right resolution (basically the same problem space as mipmaps).
        // TODO we can actually derive scale from radius. In fact, they're almost equal. I'm just
        // going to use radius for now.
        pub radius: f64,
    }

    pub struct HeightMapProperties {
        pub height: f64,
        // pub normal: Dir3,
    }
}
