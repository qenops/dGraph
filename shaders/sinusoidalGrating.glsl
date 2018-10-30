
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
   
    // Input parameters for grating
    //vec4 box = vec4(.2,.2,.4,.4);
    //vec4 box = vec4(.6,.3,.8,.8);
    vec4 box = vec4(.0,.0,1.0,1.0);
    float cycles = 35.0;
    const int direction = 0;
    float brightness = 1.0;
    float contrast = 1.0;

    // Calculate color
    const float PI = 3.1415926535897932384626433832795;
    float coord = uv[direction] - box[direction];
    // calculate normalized cycles
    float boxCycles = cycles/(box[direction+2] - box[direction]);
    // calculate bounding box
    vec2 bl = step(box.xy,uv.xy);
    vec2 tr = 1.0-step(box.zw,uv.xy);
    float shade = (bl.x*bl.y*tr.x*tr.y);
    // calculate color
    float col1 = sin((coord+.75/boxCycles)*2.0*PI*boxCycles);
    float col2 = sin((coord+.1/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col3 = sin((coord-.1/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col4 = sin((coord+.2/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col5 = sin((coord-.2/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col6 = sin((coord+.3/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col7 = sin((coord-.3/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col8 = sin((coord+.4/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col9 = sin((coord-.4/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col10 = sin((coord+.5/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float col11 = sin((coord-.5/iResolution[direction]+.75/boxCycles)*2.0*PI*boxCycles);
    float avg = (col1 + col2 + col3 + col4 + col5 + col6 + col7 + col8 + col9 + col10 + col11) / 11.0;
    float col = shade * (avg / 2.0 * contrast + (.5*brightness));
    //float col = shade * (col1/ 2.0 * contrast + (.5*brightness));
    
    // Output to screen
    fragColor = vec4(col,col,col,1.0);
}