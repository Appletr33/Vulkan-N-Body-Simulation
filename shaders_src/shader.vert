#version 450

layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 Velocity;

//push constants block
layout(push_constant) uniform Constants
{
    mat4 view;
} pushConstants;

layout(location = 0) out vec4 vColor;

void main()
{
    gl_Position = pushConstants.view * vec4(Position.x, Position.y, 0.0, 1.0);
    //gl_PointSize = 10.0 * exp2(dot(Position, Position) * -3.0);
    gl_PointSize = 2.0;
    vColor = vec4(abs(Velocity.x) * 0.1f, 0.8f, 0.5f, 1.0f);
}
