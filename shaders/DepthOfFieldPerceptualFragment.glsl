#version 330
#ifdef GL_ES
precision highp float;
#endif
in vec2 fragTexCoord;

uniform sampler2D imageTexture;
uniform sampler2D depthTexture;

uniform float focalPlaneMeters; // assumed in meters
uniform float pixelSizeMm;		// assumed in mm, def = 0.25
uniform float apertureMm;		// aperture, assumed in mm, Human pupil is between 1.5 and 8 mm

// Force location to 0 to ensure its the first output
layout(location = 0) out vec4 FragColor;


vec4 texelFetch2DClamp(const in sampler2D sampler, const in ivec2 texCoord, const in int level) {
	return texelFetch(sampler, ivec2(max(ivec2(0), min(textureSize(sampler, level) - ivec2(1), texCoord))), level);
}

vec4 computeBiCubicWeight(const float a) {
	float a2 = a*a;
	float a3 = a2*a;

	float w1 = (-a3 + 3 * a2 - 3 * a + 1) / 6;
	float w2 = (3 * a3 - 6 * a2 + 4) / 6;
	float w3 = (-3 * a3 + 3 * a2 + 3 * a + 1) / 6;
	float w4 = (a3) / 6;

	return vec4(w1, w2, w3, w4);
}

vec4 bicubicTexture(const sampler2D texture2D, const vec2 texCoord, const int level)
{
	vec2 coord = texCoord * vec2(textureSize(texture2D, level));

	vec4 horizontalWeight = computeBiCubicWeight(fract(coord.x + 0.5));
	vec4 verticalWeight = computeBiCubicWeight(fract(coord.y + 0.5));

	ivec2 index = ivec2(coord + vec2(0.5)) - ivec2(2);

	vec4 texel00 = texelFetch2DClamp(texture2D, index + ivec2(0, 0), level);
	vec4 texel01 = texelFetch2DClamp(texture2D, index + ivec2(1, 0), level);
	vec4 texel02 = texelFetch2DClamp(texture2D, index + ivec2(2, 0), level);
	vec4 texel03 = texelFetch2DClamp(texture2D, index + ivec2(3, 0), level);
	vec4 texel10 = texelFetch2DClamp(texture2D, index + ivec2(0, 1), level);
	vec4 texel11 = texelFetch2DClamp(texture2D, index + ivec2(1, 1), level);
	vec4 texel12 = texelFetch2DClamp(texture2D, index + ivec2(2, 1), level);
	vec4 texel13 = texelFetch2DClamp(texture2D, index + ivec2(3, 1), level);
	vec4 texel20 = texelFetch2DClamp(texture2D, index + ivec2(0, 2), level);
	vec4 texel21 = texelFetch2DClamp(texture2D, index + ivec2(1, 2), level);
	vec4 texel22 = texelFetch2DClamp(texture2D, index + ivec2(2, 2), level);
	vec4 texel23 = texelFetch2DClamp(texture2D, index + ivec2(3, 2), level);
	vec4 texel30 = texelFetch2DClamp(texture2D, index + ivec2(0, 3), level);
	vec4 texel31 = texelFetch2DClamp(texture2D, index + ivec2(1, 3), level);
	vec4 texel32 = texelFetch2DClamp(texture2D, index + ivec2(2, 3), level);
	vec4 texel33 = texelFetch2DClamp(texture2D, index + ivec2(3, 3), level);

	return
		texel00 * horizontalWeight.x * verticalWeight.x +
		texel01 * horizontalWeight.y * verticalWeight.x +
		texel02 * horizontalWeight.z * verticalWeight.x +
		texel03 * horizontalWeight.w * verticalWeight.x +

		texel10 * horizontalWeight.x * verticalWeight.y +
		texel11 * horizontalWeight.y * verticalWeight.y +
		texel12 * horizontalWeight.z * verticalWeight.y +
		texel13 * horizontalWeight.w * verticalWeight.y +

		texel20 * horizontalWeight.x * verticalWeight.z +
		texel21 * horizontalWeight.y * verticalWeight.z +
		texel22 * horizontalWeight.z * verticalWeight.z +
		texel23 * horizontalWeight.w * verticalWeight.z +

		texel30 * horizontalWeight.x * verticalWeight.w +
		texel31 * horizontalWeight.y * verticalWeight.w +
		texel32 * horizontalWeight.z * verticalWeight.w +
		texel33 * horizontalWeight.w * verticalWeight.w
		;
}

// converges to full pass for level == 0 (contrary to others which always do some filtering)
vec4 doubleBicubicTextureFullPass(const sampler2D texture2D, const vec2 texCoord, const float level) {
	int levelIndex = int(level);

	vec4 texel0 = levelIndex == 0 ? textureLod(texture2D, texCoord, 0.0) : bicubicTexture(texture2D, texCoord, levelIndex + 0);
	vec4 texel1 = bicubicTexture(texture2D, texCoord, levelIndex + 1);

	float levelWeight = fract(level);
	return mix(texel0, texel1, levelWeight);
}

float domainRadiusToBicubicGaussMIPLevel(const float radiusPx) {
	float supportSize = radiusPx * 2 + 1;
	float supportLog2 = log(supportSize) / log(2.0);

	float smallApprox = max(radiusPx * 0.2, 0.0);
	float largeApprox = supportLog2 - 2.5;

	float linCoef = clamp((radiusPx - 4) / (8 - 4), 0.0, 1.0);
	return mix(smallApprox, largeApprox, linCoef);
}

float getCircleOfConfusionRadius(float focusDistanceM, float targetDistanceM) {
	// should take args for distances to focal plane and blur plane and compute it properly
	// this is diameter
	float CoCpx = apertureMm * abs(targetDistanceM - focusDistanceM) / targetDistanceM;
	float CoC = CoCpx / pixelSizeMm;
	return CoC / 2;
}


void main() {
	//FragColor = doubleBicubicTextureFullPass(imageTexture, fragTexCoord, 0); return;

	float depthM = texelFetch(depthTexture, ivec2(gl_FragCoord.xy), 0).x;
	//depthM = textureLod(depthTexture, fragTexCoord, 0).x;

	float cocRadiusPx = getCircleOfConfusionRadius(focalPlaneMeters, depthM);
	float gaussLevel = domainRadiusToBicubicGaussMIPLevel(cocRadiusPx);
	
	vec4 blurredColor = doubleBicubicTextureFullPass(imageTexture, fragTexCoord, gaussLevel);

	FragColor = blurredColor;
}