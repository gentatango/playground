import numpy as np
import glfw
from OpenGL.GL import *
import time
import sys
import ctypes

# 簡略化したシェーダーコード
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

# より単純なシェーダーから始める
fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
uniform vec3 cameraPosition;
uniform vec3 cameraTarget;

// 球体の距離関数
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

// レイマーチング関数
vec4 raymarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    float tmax = 10.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;

        // 球体の距離計算
        float d = sdSphere(p, 1.0);

        // 表面に十分近づいたら
        if (d < 0.001) {
            // 法線を計算（中心的差分法）
            float eps = 0.001;
            vec3 n = normalize(vec3(
                sdSphere(p + vec3(eps, 0.0, 0.0), 1.0) - sdSphere(p - vec3(eps, 0.0, 0.0), 1.0),
                sdSphere(p + vec3(0.0, eps, 0.0), 1.0) - sdSphere(p - vec3(0.0, eps, 0.0), 1.0),
                sdSphere(p + vec3(0.0, 0.0, eps), 1.0) - sdSphere(p - vec3(0.0, 0.0, eps), 1.0)
            ));

            // 単純な照明計算
            vec3 light = normalize(vec3(1.0, 1.0, 1.0));
            float diff = max(dot(n, light), 0.0);
            vec3 color = vec3(1.0, 0.5, 0.2) * (0.3 + 0.7 * diff);

            // フォグ効果
            float fog = 1.0 - clamp(t / tmax, 0.0, 1.0);
            color = mix(vec3(0.05, 0.05, 0.2), color, fog);

            return vec4(color, 1.0);
        }

        // 距離だけ前進
        t += d;

        // 最大距離を超えたら終了
        if (t > tmax) break;
    }

    // 背景グラデーション
    return vec4(0.05, 0.05, 0.2, 1.0);
}

void main() {
    // 正規化スクリーン座標（-1 から 1）
    vec2 uv = (gl_FragCoord.xy / resolution.xy) * 2.0 - 1.0;
    // アスペクト比補正
    uv.x *= resolution.x / resolution.y;

    // カメラの設定
    vec3 ro = cameraPosition;
    vec3 ta = cameraTarget;

    // カメラの座標系を構築
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));

    // レイ方向
    vec3 rd = normalize(ww + uv.x * uu + uv.y * vv);

    // レイマーチング実行
    vec4 color = raymarch(ro, rd);

    // ガンマ補正
    color.rgb = pow(color.rgb, vec3(0.4545));

    // 最終出力
    FragColor = color;
}
"""

def create_window():
    if not glfw.init():
        print("GLFWの初期化に失敗しました")
        return None

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    if sys.platform == 'darwin':
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(800, 600, "3Dレンダリングテスト", None, None)
    if not window:
        print("ウィンドウの作成に失敗しました")
        glfw.terminate()
        return None

    glfw.make_context_current(window)
    return window

def main():
    window = create_window()
    if not window:
        return

    # OpenGLのバージョンを表示
    print(f"OpenGL バージョン: {glGetString(GL_VERSION).decode('utf-8')}")
    print(f"GLSL バージョン: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")

    # 全画面を覆う四角形の頂点
    vertices = np.array([
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,

        -1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
        -1.0,  1.0, 0.0
    ], dtype=np.float32)

    # VAO、VBOの設定
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

    # シェーダーのコンパイルとリンク
    vertex = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex, vertex_shader)
    glCompileShader(vertex)

    if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
        print(f"頂点シェーダーエラー: {glGetShaderInfoLog(vertex).decode('utf-8')}")
        return

    fragment = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment, fragment_shader)
    glCompileShader(fragment)

    if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
        print(f"フラグメントシェーダーエラー: {glGetShaderInfoLog(fragment).decode('utf-8')}")
        return

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex)
    glAttachShader(shader_program, fragment)
    glLinkProgram(shader_program)

    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        print(f"シェーダープログラムエラー: {glGetProgramInfoLog(shader_program).decode('utf-8')}")
        return

    glDeleteShader(vertex)
    glDeleteShader(fragment)

    # Uniform変数の場所
    resolution_loc = glGetUniformLocation(shader_program, "resolution")
    time_loc = glGetUniformLocation(shader_program, "time")
    camera_pos_loc = glGetUniformLocation(shader_program, "cameraPosition")
    camera_target_loc = glGetUniformLocation(shader_program, "cameraTarget")

    # メインループ
    start_time = time.time()
    fps_count = 0
    fps_time = start_time

    print("レンダリング開始...")

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # FPS計測
        fps_count += 1
        current_time = time.time()
        if current_time - fps_time >= 1.0:
            print(f"FPS: {fps_count / (current_time - fps_time):.1f}")
            fps_count = 0
            fps_time = current_time

        # ウィンドウサイズ
        width, height = glfw.get_window_size(window)
        glViewport(0, 0, width, height)

        # 時間
        t = current_time - start_time

        # カメラ設定
        radius = 3.0
        cam_x = radius * np.sin(t * 0.5)
        cam_z = radius * np.cos(t * 0.5)
        cam_y = 0.5 * np.sin(t * 0.3)

        camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 描画準備
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)

        # Uniform変数設定
        glUniform2f(resolution_loc, width, height)
        glUniform1f(time_loc, t)
        glUniform3fv(camera_pos_loc, 1, camera_pos)
        glUniform3fv(camera_target_loc, 1, camera_target)

        # 描画
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glfw.swap_buffers(window)

        # ESCキーで終了
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    # 終了処理
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(shader_program)

    glfw.terminate()
    print("プログラム終了")

if __name__ == "__main__":
    main()