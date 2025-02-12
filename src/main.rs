use bevy::{
    hierarchy::ReportHierarchyIssue,
    math::{dvec3, DVec3},
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
    window::CursorGrabMode,
};
use bevy_enhanced_input::prelude::*;
use big_space::prelude::*;
use controls::*;
use ico_mesh::*;

type P = i64;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin,
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

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((SpeedText, Text::new("Speed: 0m/s")));
            parent.spawn(Text::new("Press `Tab` to toggle cursor capture"));
        });
    commands.spawn_big_space(Grid::<P>::new(1e-3, 0.0), |grid| {
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
            let mesh = face.mesh();
            let (cell, offset) = grid.grid().translation_to_grid(mesh.offset);
            grid.spawn_spatial((
                Mesh3d(meshes.add(mesh.mesh)),
                Wireframe,
                cell,
                Transform::from_translation(offset),
                BoundingSphere(mesh.radius),
                face,
            ));
        }
    });
}

// If the floating origin gets close to us, we want to add a marker that says we need to be
// subdivided. But how do we know if we're already subdivided? We create another marker to track
// that and keep it in sync. That means we need a system to sync the "needs subdivision" marker and
// one to sync the "is subdivided" marker. We also need a system to subdivide faces with the "needs
// subdivision" marker and no "is subdivided" marker, and one to remove the face's children when it
// has the "is subdivided" marker and no "needs subdivision" marker.

#[derive(Component)]
struct NeedsSubdivision;

#[derive(Component)]
struct IsSubdivided([Entity; 4]);

fn mark_for_subdivision(
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

fn subdivide_faces(
    faces: Query<
        (Entity, &SubFace, &BoundingSphere),
        (With<NeedsSubdivision>, Without<IsSubdivided>),
    >,
    grid_root: Single<(Entity, &Grid<P>), With<BigSpace>>,
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
            let mesh = face.mesh();
            let (cell, offset) = grid_root.translation_to_grid(mesh.offset);
            commands
                .spawn((
                    Mesh3d(meshes.add(mesh.mesh)),
                    Wireframe,
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

fn recombine_faces(
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
struct BoundingSphere(f32);

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
    use bevy::math::FloatOrd;

    use super::*;

    pub fn icosphere(radius: f64) -> [SubFace; 20] {
        const PHI: f64 = 1.618033988749894848204586834365638117720309179805762862135448622;
        let vertices = [
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
        ]
        .map(|x| x.normalize());
        [
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
        ]
        .map(|face| SubFace {
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

    #[derive(Clone, Copy)]
    pub struct IcoFace {
        pub vertices: [usize; 3],
    }

    impl IcoFace {
        pub fn new(a: usize, b: usize, c: usize) -> Self {
            Self {
                vertices: [a, b, c],
            }
        }
    }

    /// This represents any sub-face of arbitrary order of an icosphere. A 0th-order sub-face of e.g.
    /// face 0 is face 0. Each face has 4 1st order sub-faces, 16 2nd order sub-faces, and so on.
    #[derive(Component, Clone, Copy)]
    pub struct SubFace {
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

        pub fn mesh(self) -> BigMesh {
            let vertices = self
                .coords
                .into_iter()
                .map(|x| x.position)
                .collect::<Vec<_>>();
            let faces = [[0, 1, 2]];
            let vertices = std::cell::Cell::new(vertices);
            let subdivide = |[a, b, c]: [usize; 3]| {
                let mut my_vertices = vertices.take();
                my_vertices.push(my_vertices[a].midpoint(my_vertices[b]).normalize());
                let ab = my_vertices.len() - 1;
                my_vertices.push(my_vertices[b].midpoint(my_vertices[c]).normalize());
                let bc = my_vertices.len() - 1;
                my_vertices.push(my_vertices[c].midpoint(my_vertices[a]).normalize());
                let ca = my_vertices.len() - 1;
                vertices.set(my_vertices);
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            };
            let faces = faces
                .into_iter()
                .flat_map(subdivide)
                .flat_map(subdivide)
                .flat_map(|x| x.map(|x| x as u32)) // Flatten [[usize; 3]] -> [u32]
                .collect::<Vec<_>>();
            let vertices = vertices.into_inner();
            let offset = vertices.iter().sum::<DVec3>() * self.radius / (vertices.len() as f64);
            let uvs = vec![Vec2::ZERO; vertices.len()];
            let vertices = vertices
                .iter()
                .map(|x| (self.radius * x - offset).as_vec3())
                .collect::<Vec<_>>();
            let FloatOrd(radius) = vertices.iter().map(|x| FloatOrd(x.length())).max().unwrap();

            let mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
                .with_inserted_indices(Indices::U32(faces))
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
                .with_computed_normals();

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
}
