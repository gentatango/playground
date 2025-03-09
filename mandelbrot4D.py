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

// 4次元複素数の演算
vec4 multiply_quaternion(vec4 a, vec4 b) {
    return vec4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

// 4次元複素数のノルム
float norm_squared(vec4 q) {
    return q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
}

void main() {
    // 正規化された座標
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);

    // ズームと中心点を適用
    uv = uv / zoom + center;

    // 時間によって変わる追加の次元パラメータ
    float z_param = sin(time * 0.2) * 0.2;
    float w_param = cos(time * 0.15) * 0.2;

    // 4次元複素数を初期化
    vec4 c = vec4(uv.x, uv.y, z_param, w_param);
    vec4 z = vec4(0.0, 0.0, 0.0, 0.0);

    // 反復回数
    int max_iter = 100;
    int iter = 0;

    // マンデルブロ集合の計算
    for (iter = 0; iter < max_iter; iter++) {
        z = multiply_quaternion(z, z) + c;

        if (norm_squared(z) > 4.0) {
            break;
        }
    }

    // カラー計算
    if (iter == max_iter) {
        // マンデルブロ集合内部は黒
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        // 外部はカラフルに
        float t = float(iter) / float(max_iter);

        // 色のグラデーション
        vec3 color1 = vec3(0.1, 0.8, 0.9);
        vec3 color2 = vec3(0.9, 0.1, 0.2);
        vec3 color3 = vec3(0.1, 0.9, 0.2);

        // 平滑なカラーリング
        float angle = 6.28318 * (t + time * 0.1);
        vec3 color = color1 * (sin(angle) * 0.5 + 0.5) +
                     color2 * (sin(angle + 2.09) * 0.5 + 0.5) +
                     color3 * (sin(angle + 4.18) * 0.5 + 0.5);

        gl_FragColor = vec4(color, 1.0);
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

    # インデックスデータ
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

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
    pygame.display.set_caption("4D Mandelbrot Set")

    # リセット前にバージョン情報を表示
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
    print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")

    program, vbo, position_loc = init_gl()

    # ユニフォーム変数の位置を取得
    resolution_loc = glGetUniformLocation(program, "resolution")
    time_loc = glGetUniformLocation(program, "time")
    center_loc = glGetUniformLocation(program, "center")
    zoom_loc = glGetUniformLocation(program, "zoom")

    # 初期パラメータ
    center = [0.0, 0.0]
    zoom = 1.0
    zoom_speed = 1.01

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

        # 時間を更新
        current_time = time.time() - start_time

        # シェーダープログラムを使用
        glUseProgram(program)

        # ユニフォーム変数を設定
        glUniform2f(resolution_loc, WIDTH, HEIGHT)
        glUniform1f(time_loc, current_time)
        glUniform2f(center_loc, center[0], center[1])
        glUniform1f(zoom_loc, zoom)

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