#version 450

layout(location = 0) out vec4 FragColor;
layout(location = 0) in vec4 vColor;

float falloff(vec2 point_coord)
{
    vec2 center_dist = point_coord - 0.5;
    float dist_sqr = dot(center_dist, center_dist);
    return exp2(-dist_sqr * 15.0);
}

void main()
{
    //vec2 coord = gl_PointCoord.xy;
    //FragColor = vec4(falloff(coord) * vColor.rgb, vColor.a);
    FragColor = vec4(vColor.rgb, vColor.a);
}
