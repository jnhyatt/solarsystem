use bevy::{
    hierarchy::ReportHierarchyIssue,
    math::{dvec3, DMat3, DVec2, DVec3},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
    window::CursorGrabMode,
};
use bevy_enhanced_input::prelude::*;
use big_space::prelude::*;
use controls::*;
use ico_mesh::*;
use rand_pcg::Pcg32;

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
        DirectionalLight::default(),
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
    use super::*;
    use bevy::math::FloatOrd;

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
        let vertices = VERTICES.map(|x| x.normalize());
        FACES.map(|face| SubFace {
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
        })
    }

    #[derive(Clone, Copy)]
    pub struct FaceCoord {
        /// Normalized position on the icosphere.
        pub position: DVec3,
        pub barycentric_coords: [f64; 3],
    }

    impl FaceCoord {
        fn midpoint(self, right: Self) -> Self {
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
            let unit_sphere = self
                .coords
                .into_iter()
                .map(|x| x.position)
                .collect::<Vec<_>>();
            let faces = [[0, 1, 2]];
            let unit_sphere = std::cell::Cell::new(unit_sphere);
            let subdivide = |[a, b, c]: [usize; 3]| {
                let mut my_vertices = unit_sphere.take();
                my_vertices.push(my_vertices[a].midpoint(my_vertices[b]).normalize());
                let ab = my_vertices.len() - 1;
                my_vertices.push(my_vertices[b].midpoint(my_vertices[c]).normalize());
                let bc = my_vertices.len() - 1;
                my_vertices.push(my_vertices[c].midpoint(my_vertices[a]).normalize());
                let ca = my_vertices.len() - 1;
                unit_sphere.set(my_vertices);
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            };
            let faces = faces
                .into_iter()
                .flat_map(subdivide)
                .flat_map(subdivide)
                .flat_map(subdivide)
                .flat_map(|x| x.map(|x| x as u32)) // Flatten [[usize; 3]] -> [u32]
                .collect::<Vec<_>>();
            let unit_sphere = unit_sphere.into_inner();
            let vertices = unit_sphere
                .into_iter()
                .map(|x| {
                    let props = height_map.properties(&Location {
                        normalized_position: x,
                        face: self.face,
                        radius: self.radius,
                    });
                    x * (self.radius + props.height)
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
        /// The mesh in single precision, centered at the origin.
        pub mesh: Mesh,
        /// The offset from the origin. Before the mesh was recentered, this was the average of its
        /// vertex positions.
        pub offset: DVec3,
        /// The bounding radius of the mesh.
        pub radius: f32,
    }

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
