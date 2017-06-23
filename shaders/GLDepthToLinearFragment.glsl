#version 330
#ifdef GL_ES
precision highp float;
#endif

in vec2 fragTexCoord;

uniform sampler2D glDepthTexture2D;
uniform vec2 inputRange;
uniform vec2 outputRange;
uniform mat4 projectionMatrix;

layout(location = 0) out float linearDepth;


float glDepthToLinearDepth(const float glDepth, const float near, const float far) {
	float clipSpaceDepth = 2.0 * glDepth - 1.0;
	return 2.0 * near * far / (far + near - clipSpaceDepth * (far - near));
}

float getNear(const mat4 projectionMatrix) {
	return projectionMatrix[3][2] / (projectionMatrix[2][2] - 1);
}

float getFar(const mat4 projectionMatrix) {
	return projectionMatrix[3][2] / (projectionMatrix[2][2] + 1);
}

void main() {
	float glDepth = texture(glDepthTexture2D, fragTexCoord).r;
	float near = getNear(projectionMatrix);
	float far = getFar(projectionMatrix);
	linearDepth = glDepthToLinearDepth(glDepth, near, far);
	//linearDepth = linearDepth * 1e-10 + 0.5;

	if (linearDepth <= 0) {
		linearDepth = 1e20;
		return;
	}

	linearDepth = (linearDepth - inputRange.x) / (inputRange.y - inputRange.x) * (outputRange.y - outputRange.x) + outputRange.x;
}
