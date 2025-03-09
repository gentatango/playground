import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import time

# ウィンドウのサイズを設定
WIDTH, HEIGHT = 800, 600

# フラグメントシェーダーのソースコード
fragment_shader = """
#version 120

uniform vec2 resolution;
uniform float time;
uniform vec2 center;
uniform float zoom;
uniform float power;

void main() {
    // 正規化された座標
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);

    // ズームと中心点を適用
    uv = uv / zoom + center;

    // レイマーチングのパラメータ
    vec3 ro = vec3(0.0, 0.0, -2.5); // レイの原点
    vec3 rd = normalize(vec3(uv.x, uv.y, 1.0)); // レイの方向

    // アニメーション用の回転行列
    float s = sin(time * 0.2);
    float c = cos(time * 0.2);
    mat3 rot_x = mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
    s = sin(time * 0.15);
    c = cos(time * 0.15);
    mat3 rot_y = mat3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);

    // レイの方向のみを回転させるのではなく、シーン全体を回転させる
    // オブジェクトの方を回転させるので、逆回転を適用
    mat3 rot = rot_y * rot_x;
    mat3 inv_rot = transpose(rot); // 直交行列の逆行列は転置行列

    // レイの原点と方向を回転させる
    ro = inv_rot * ro;
    rd = inv_rot * rd;

    // マンデルバルブのパラメータ
    float n = mix(6.0, 12.0, (sin(time * 0.1) * 0.5 + 0.5)); // 次数を時間によって変更

    // レイマーチング
    float t = 0.0;
    float min_dist = 1000.0;
    int max_steps = 100;
    float escape_radius = 2.0;
    float surface_dist = 0.001;
    bool hit = false;
    int steps = 0;

    for (int i = 0; i < max_steps; i++) {
        vec3 p = ro + rd * t;

        // マンデルバルブの距離推定関数
        vec3 z = p;
        float dr = 1.0;
        float r = 0.0;

        for (int j = 0; j < 10; j++) {
            r = length(z);

            if (r > escape_radius) break;

            // 極座標への変換
            float theta = acos(z.z / r);
            float phi = atan(z.y, z.x);
            dr = pow(r, n - 1.0) * n * dr + 1.0;

            // rのべき乗
            float zr = pow(r, n);
            // 角度の乗算
            theta = theta * n;
            phi = phi * n;

            // 直交座標へ戻す
            z = zr * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            z += p; // 元の点を加算
        }

        float dist = 0.5 * log(r) * r / dr;

        if (dist < surface_dist) {
            hit = true;
            break;
        }

        t += max(dist * 0.5, surface_dist);
        min_dist = min(min_dist, dist);
        steps = i;

        if (t > 10.0) break;
    }

    // カラー計算
    if (hit) {
        // ヒットした場所に基づいて色を計算
        vec3 p = ro + rd * t;

        // 空間座標に基づく色
        vec3 base_color = 0.5 + 0.5 * cos(time * 0.2 + p.z + vec3(0, 2, 4));

        // 影を計算する簡易的な手法（アンビエントオクルージョン）
        float ao = float(steps) / float(max_steps);
        ao = 1.0 - ao;

        gl_FragColor = vec4(base_color * ao, 1.0);
    } else {
        // 背景色
        float d = min_dist / 2.0;
        d = clamp(d, 0.0, 1.0);

        // グラデーション背景
        vec3 bg1 = vec3(0.1, 0.1, 0.2);
        vec3 bg2 = vec3(0.0, 0.0, 0.0);
        vec3 bg = mix(bg1, bg2, length(uv));

        gl_FragColor = vec4(bg, 1.0);
    }
}
"""

# 頂点シェーダーのソースコード
vertex_shader = """
#version 120

attribute vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# OpenGLの初期化とシェーダーのコンパイル
def init_gl():
    # シェーダーのコンパイル
    vertex = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    fragment = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vertex, fragment)

    # 頂点データ（スクリーンサイズのクワッド）
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0
    ], dtype=np.float32)

    # VBOを作成して頂点データをアップロード
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # OpenGL 2.1では、VAOの代わりに属性ポインタを設定
    glUseProgram(program)
    position_loc = glGetAttribLocation(program, 'position')

    return program, vbo, position_loc

def main():
    pygame.init()
    display = (WIDTH, HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Mandelbulb")

    # リセット前にバージョン情報を表示
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
    print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")

    program, vbo, position_loc = init_gl()

    # ユニフォーム変数の位置を取得
    resolution_loc = glGetUniformLocation(program, "resolution")
    time_loc = glGetUniformLocation(program, "time")
    center_loc = glGetUniformLocation(program, "center")
    zoom_loc = glGetUniformLocation(program, "zoom")
    power_loc = glGetUniformLocation(program, "power")

    # 初期パラメータ
    center = [0.0, 0.0]
    zoom = 1.0
    zoom_speed = 1.01
    power = 8.0

    # メインループ
    start_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    zoom *= zoom_speed
                elif event.key == pygame.K_MINUS:
                    zoom /= zoom_speed
                elif event.key == pygame.K_UP:
                    center[1] -= 0.1 / zoom
                elif event.key == pygame.K_DOWN:
                    center[1] += 0.1 / zoom
                elif event.key == pygame.K_LEFT:
                    center[0] += 0.1 / zoom
                elif event.key == pygame.K_RIGHT:
                    center[0] -= 0.1 / zoom
                elif event.key == pygame.K_r:
                    center = [0.0, 0.0]
                    zoom = 1.0
                elif event.key == pygame.K_1:
                    power = 2.0
                elif event.key == pygame.K_2:
                    power = 4.0
                elif event.key == pygame.K_3:
                    power = 8.0

        # 時間を更新
        current_time = time.time() - start_time

        # シェーダープログラムを使用
        glUseProgram(program)

        # ユニフォーム変数を設定
        glUniform2f(resolution_loc, WIDTH, HEIGHT)
        glUniform1f(time_loc, current_time)
        glUniform2f(center_loc, center[0], center[1])
        glUniform1f(zoom_loc, zoom)
        glUniform1f(power_loc, power)

        # 頂点属性を有効化
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(position_loc)
        glVertexAttribPointer(position_loc, 2, GL_FLOAT, GL_FALSE, 8, None)

        # 画面をクリア
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 描画
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

        # 頂点属性を無効化
        glDisableVertexAttribArray(position_loc)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # 画面を更新
        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()