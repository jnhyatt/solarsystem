use bevy::{
    asset::RenderAssetUsages,
    math::{dvec3, DVec3},
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::mesh::PrimitiveTopology,
    window::CursorGrabMode,
};
use bevy_enhanced_input::prelude::*;
use big_space::{
    commands::BigSpaceCommands,
    floating_origins::FloatingOrigin,
    grid::{cell::GridCell, local_origin::Grids, Grid},
    plugin::BigSpacePlugin,
};

#[derive(Clone, Copy)]
struct FaceCoord {
    /// Normalized position on the icosphere.
    position: DVec3,
    barycentric_coords: [f64; 3],
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
struct IcoFace {
    vertices: [usize; 3],
}

impl IcoFace {
    fn new(a: usize, b: usize, c: usize) -> Self {
        Self {
            vertices: [a, b, c],
        }
    }
}

/// This represents any sub-face of arbitrary order of an icosphere. A 0th-order sub-face of e.g.
/// face 0 is face 0. Each face has 4 1st order sub-faces, 16 2nd order sub-faces, and so on.
struct SubFace {
    face: IcoFace,
    coords: [FaceCoord; 3],
}

impl SubFace {
    fn subdivide(self) -> [Self; 4] {
        let ab = self.coords[0].midpoint(self.coords[1]);
        let bc = self.coords[1].midpoint(self.coords[2]);
        let ca = self.coords[2].midpoint(self.coords[0]);
        [
            Self {
                face: self.face,
                coords: [self.coords[0], ab, ca],
            },
            Self {
                face: self.face,
                coords: [self.coords[1], bc, ab],
            },
            Self {
                face: self.face,
                coords: [self.coords[2], ca, bc],
            },
            Self {
                face: self.face,
                coords: [ab, bc, ca],
            },
        ]
    }

    fn mesh(self) -> BigMesh {
        let offset = self.coords.into_iter().map(|x| x.position).sum::<DVec3>() / 3.0;
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            self.coords
                .into_iter()
                .map(|x| (x.position - offset).as_vec3())
                .collect::<Vec<_>>(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![Vec2::ZERO; 3])
        .with_computed_flat_normals();
        BigMesh { mesh, offset }
    }
}

#[derive(Clone)]
struct BigMesh {
    mesh: Mesh,
    offset: DVec3,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.build().disable::<TransformPlugin>(),
            WireframePlugin,
            EnhancedInputPlugin,
            BigSpacePlugin::<i32>::default(),
        ))
        .add_input_context::<SpaceshipControls>()
        .add_observer(translation_input)
        .add_observer(rotation_input)
        .add_observer(adjust_speed)
        .add_observer(
            |_: Trigger<Started<ToggleCursorCapture>>, mut window: Single<&mut Window>| {
                let was_captured = window.cursor_options.grab_mode != CursorGrabMode::None;
                window.cursor_options.grab_mode = if was_captured {
                    CursorGrabMode::None
                } else {
                    CursorGrabMode::Locked
                };
                window.cursor_options.visible = was_captured;
            },
        )
        .add_systems(Startup, setup)
        .add_systems(Update, adjust_speed_text)
        .run();
}

#[derive(Component)]
struct SpeedText;

fn translation_input(
    input: Trigger<Fired<Translation>>,
    mut player: Query<(&mut Transform, &mut GridCell<i32>, &Speed)>,
    grids: Grids<i32>,
    time: Res<Time>,
) {
    let (mut transform, mut grid_cell, &Speed(speed)) = player.get_mut(input.entity()).unwrap();
    let movement = transform.transform_point(input.value * speed * time.delta_secs());
    let grid = grids.parent_grid(input.entity()).unwrap();
    let (cell_offset, movement) = grid.translation_to_grid(movement);
    *grid_cell += cell_offset;
    transform.translation += movement;
}

fn rotation_input(
    input: Trigger<Fired<Rotation>>,
    mut player: Query<&mut Transform>,
    time: Res<Time>,
) {
    let mut player = player.get_mut(input.entity()).unwrap();
    player.rotation *= Quat::from_scaled_axis(input.value * time.delta_secs());
}

#[derive(Component)]
struct Speed(f32);

fn adjust_speed(input: Trigger<Fired<AdjustSpeed>>, mut player: Query<&mut Speed>) {
    let mut speed = player.get_mut(input.entity()).unwrap();
    speed.0 = speed.0 * (input.value * 0.1).exp();
}

fn adjust_speed_text(
    speed: Option<Single<&Speed, Changed<Speed>>>,
    mut text: Single<&mut Text, With<SpeedText>>,
) {
    let Some(speed) = speed else {
        return;
    };
    ***text = format!("Speed: {}m/s", speed.0);
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
    commands.spawn_big_space(Grid::<i32>::new(1.0, 0.0), |grid| {
        grid.spawn_spatial((
            SpaceshipControls,
            Speed(1.0),
            Camera3d::default(),
            Projection::Perspective(PerspectiveProjection {
                near: 1e-6,
                ..default()
            }),
            Transform::from_xyz(0.0, 0.0, 10.0),
            FloatingOrigin,
        ));

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
        let faces = [
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
        })
        .map(|sub_face| sub_face.mesh());
        for mesh in faces {
            let (cell, offset) = grid.grid().translation_to_grid(mesh.offset);
            let mesh = meshes.add(mesh.mesh);
            grid.spawn_spatial((
                Mesh3d(mesh),
                Wireframe,
                cell,
                Transform::from_translation(offset),
            ));
        }
    });
}

#[derive(Debug, InputAction)]
#[input_action(output = Vec3)]
struct Translation;

#[derive(Debug, InputAction)]
#[input_action(output = Vec3)]
struct Rotation;

#[derive(Debug, InputAction)]
#[input_action(output = f32)]
struct AdjustSpeed;

#[derive(Component)]
struct SpaceshipControls;

#[derive(Debug, InputAction)]
#[input_action(output = bool)]
struct ToggleCursorCapture;

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
        ctx.bind::<Rotation>().to((
            Input::mouse_motion().with_modifiers((
                SwizzleAxis::YXZ,
                Negate::all(),
                Scale::splat(0.1),
            )),
            KeyCode::KeyQ.with_modifiers(SwizzleAxis::YZX),
            KeyCode::KeyE.with_modifiers((SwizzleAxis::YZX, Negate::all())),
        ));
        ctx.bind::<AdjustSpeed>()
            .to(Input::mouse_wheel().with_modifiers((SwizzleAxis::YXZ, Scale::splat(2.4))));
        ctx.bind::<ToggleCursorCapture>().to(KeyCode::Tab);
        ctx
    }
}
