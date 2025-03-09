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
uniform int iterations;

const int MAX_MARCHING_STEPS = 100;
const float MIN_DIST = 0.001;
const float MAX_DIST = 20.0;
const float EPSILON = 0.0001;

// 立方体の距離関数
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// メンガースポンジの距離関数 - 修正版
float mengerSponge(vec3 p, int iterations) {
    float d = sdBox(p, vec3(1.0));

    float s = 1.0;
    for (int i = 0; i < 5; i++) {
        if (i >= iterations) break;

        // スケールを調整
        vec3 a = mod(p * s, 2.0) - 1.0;
        s *= 3.0;

        vec3 r = abs(1.0 - 3.0 * abs(a));

        float da = max(r.x, r.y);
        float db = max(r.y, r.z);
        float dc = max(r.z, r.x);
        float c = (min(da, min(db, dc)) - 1.0) / s;

        d = max(d, c);
    }

    return d;
}

// レイマーチング関数
float raymarch(vec3 ro, vec3 rd, int iterations) {
    float depth = MIN_DIST;
    float dist;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        vec3 p = ro + depth * rd;
        dist = mengerSponge(p, iterations);

        if (dist < EPSILON) {
            return depth;
        }

        depth += dist * 0.5; // 距離に係数を掛けてステップを小さくする

        if (depth >= MAX_DIST) {
            return MAX_DIST;
        }
    }

    return MAX_DIST;
}

// 法線の計算
vec3 estimateNormal(vec3 p, int iterations) {
    float h = 0.0001;
    vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * mengerSponge(p + k.xyy * h, iterations) +
        k.yxy * mengerSponge(p + k.yxy * h, iterations) +
        k.yyx * mengerSponge(p + k.yyx * h, iterations) +
        k.xxx * mengerSponge(p + k.xxx * h, iterations)
    );
}

void main() {
    // 正規化された座標
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / min(resolution.x, resolution.y);

    // ズームと中心点を適用
    uv = uv / zoom + vec2(center.x, center.y);

    // カメラの設定
    vec3 ro = vec3(0.0, 0.0, 3.0); // カメラをZ軸正方向に移動
    vec3 target = vec3(0.0, 0.0, 0.0);
    vec3 up = vec3(0.0, 1.0, 0.0);

    // カメラの回転（時間ベース）
    float angle = time * 0.3;
    float radius = 3.0;
    ro.x = sin(angle) * radius;
    ro.z = cos(angle) * radius;

    // カメラの行列を設定
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(forward, up));
    up = cross(right, forward);

    // レイの方向
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);

    // レイマーチング
    float depth = raymarch(ro, rd, iterations);

    // 色の計算
    vec3 color;

    if (depth < MAX_DIST) {
        // ヒットポイントを計算
        vec3 p = ro + rd * depth;

        // 法線を計算
        vec3 normal = estimateNormal(p, iterations);

        // ライトの方向
        vec3 light1 = normalize(vec3(sin(time * 0.7), 0.6, cos(time * 0.7)));
        vec3 light2 = normalize(vec3(-sin(time * 0.5), 0.4, -cos(time * 0.5)));

        // アンビエント
        float ambient = 0.2;

        // ディフューズ
        float diffuse1 = max(0.0, dot(normal, light1));
        float diffuse2 = max(0.0, dot(normal, light2));

        // スペキュラー
        vec3 ref1 = reflect(-light1, normal);
        float spec1 = pow(max(0.0, dot(ref1, -rd)), 16.0);

        vec3 ref2 = reflect(-light2, normal);
        float spec2 = pow(max(0.0, dot(ref2, -rd)), 16.0);

        // フレネル効果
        float fresnel = pow(1.0 - max(0.0, dot(normal, -rd)), 4.0);

        // AOの簡易計算
        float ao = 1.0 - float(iterations) / 5.0;

        // 色の合成
        vec3 baseColor = mix(vec3(0.2, 0.4, 0.8), vec3(0.8, 0.6, 0.2), fresnel);

        color = baseColor * ambient +
                vec3(0.9, 0.8, 0.7) * diffuse1 * 0.7 +
                vec3(0.3, 0.4, 0.8) * diffuse2 * 0.5 +
                vec3(1.0, 0.9, 0.8) * spec1 * 0.8 +
                vec3(0.5, 0.6, 1.0) * spec2 * 0.3 +
                vec3(0.4, 0.5, 0.6) * fresnel * 0.4;

        // AOを適用
        color *= ao;
    } else {
        // 背景色（グラデーション）
        color = mix(
            vec3(0.05, 0.05, 0.1),
            vec3(0.0, 0.0, 0.0),
            min(1.0, length(uv))
        );
    }

    // ガンマ補正
    color = pow(color, vec3(0.4545));

    gl_FragColor = vec4(color, 1.0);
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
    pygame.display.set_caption("3D Menger Sponge")

    # リセット前にバージョン情報を表示
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
    print(f"GLSL Version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")

    program, vbo, position_loc = init_gl()

    # ユニフォーム変数の位置を取得
    resolution_loc = glGetUniformLocation(program, "resolution")
    time_loc = glGetUniformLocation(program, "time")
    center_loc = glGetUniformLocation(program, "center")
    zoom_loc = glGetUniformLocation(program, "zoom")
    iterations_loc = glGetUniformLocation(program, "iterations")

    # 初期パラメータ
    center = [0.0, 0.0]
    zoom = 1.0
    zoom_speed = 1.05
    iterations = 3  # 初期反復回数（パフォーマンスと複雑さのバランス）

    # メインループ
    start_time = time.time()
    last_time = start_time
    frame_count = 0
    fps = 0

    running = True
    while running:
        current_time = time.time()
        frame_count += 1

        # 1秒ごとにFPSを更新
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time

            # FPS表示をウィンドウのタイトルに更新
            pygame.display.set_caption(f"3D Menger Sponge - FPS: {fps} - Iterations: {iterations}")

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
                    iterations = max(1, iterations - 1)
                elif event.key == pygame.K_2:
                    iterations = min(5, iterations + 1)
                elif event.key == pygame.K_p:
                    # 一時停止/再開
                    paused = True
                    pause_time = current_time - start_time
                    while paused:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.QUIT:
                                paused = False
                                running = False
                            elif pause_event.type == pygame.KEYDOWN:
                                if pause_event.key == pygame.K_p or pause_event.key == pygame.K_ESCAPE:
                                    paused = False
                        pygame.time.wait(100)
                    start_time = current_time - pause_time

        # 時間を更新
        elapsed_time = time.time() - start_time

        # シェーダープログラムを使用
        glUseProgram(program)

        # ユニフォーム変数を設定
        glUniform2f(resolution_loc, WIDTH, HEIGHT)
        glUniform1f(time_loc, elapsed_time)
        glUniform2f(center_loc, center[0], center[1])
        glUniform1f(zoom_loc, zoom)
        glUniform1i(iterations_loc, iterations)

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