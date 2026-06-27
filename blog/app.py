import re
import sqlite3
import os
from functools import wraps

import bcrypt
import bleach
import markdown as md
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
DATABASE = "/data/blog.db"


# ── Markdown rendering ────────────────────────────────────────────────────────

ALLOWED_TAGS = [
    "a", "abbr", "b", "blockquote", "br", "caption", "cite", "code", "col",
    "colgroup", "dd", "del", "details", "dfn", "div", "dl", "dt", "em",
    "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
    "i", "img", "ins", "kbd", "li", "mark", "ol", "p", "pre", "q",
    "s", "samp", "small", "span", "strong", "sub", "summary", "sup",
    "table", "tbody", "td", "tfoot", "th", "thead", "time", "tr", "u",
    "ul", "var",
]
ALLOWED_ATTRS = {
    "a": ["href", "title", "rel"],
    "abbr": ["title"],
    "col": ["span", "width"],
    "colgroup": ["span", "width"],
    "img": ["src", "alt", "title", "width", "height"],
    "ol": ["start", "type"],
    "td": ["colspan", "rowspan", "align"],
    "th": ["colspan", "rowspan", "align", "scope"],
    "time": ["datetime"],
    "li": ["value"],
    "pre": ["class"],
    "code": ["class"],
    "span": ["class"],
    "div": ["class"],
    "table": ["class"],
    "details": ["open"],
}


def render_markdown(text):
    """Convert Markdown text to safe HTML."""
    if not text:
        return ""
    html = md.markdown(
        text,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "toc",
            "sane_lists",
        ],
    )
    # Sanitize with bleach to prevent XSS
    safe_html = bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        strip=True,
    )
    return safe_html


def strip_html(text):
    """Strip HTML tags from a string, returning plain text."""
    return bleach.clean(text, tags=[], strip=True) if text else ""


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                post_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (post_id) REFERENCES posts(id)
            );
        """)


# Initialize database on module load (so gunicorn workers pick it up)
init_db()


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Template routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/posts/<int:post_id>")
def post_page(post_id):
    return render_template("post.html", post_id=post_id)


# ── API: Auth ────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    email = data.get("email", "").strip()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, hashed.decode("utf-8")),
            )
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409

    return jsonify({"message": "User registered successfully"}), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    email = data.get("email", "").strip()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    with get_db() as db:
        user = db.execute(
            "SELECT id, email, password FROM users WHERE email = ?", (email,)
        ).fetchone()

    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        return jsonify({"error": "Invalid email or password"}), 401

    session["user_id"] = user["id"]
    session["email"] = user["email"]
    return jsonify({"message": "Logged in successfully", "email": user["email"]}), 200


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200


# ── API: Posts ───────────────────────────────────────────────────────────────

@app.route("/api/posts", methods=["GET"])
def api_list_posts():
    with get_db() as db:
        rows = db.execute("""
            SELECT p.id, p.title, p.content, p.user_id, p.created_at, u.email as author
            FROM posts p JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
        """).fetchall()
    posts = []
    for r in rows:
        p = dict(r)
        p["content_html"] = render_markdown(p["content"])
        p["content_preview"] = strip_html(p["content_html"])[:300]
        posts.append(p)
    return jsonify(posts), 200


@app.route("/api/posts", methods=["POST"])
@require_auth
def api_create_post():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    if not title or not content:
        return jsonify({"error": "Title and content are required"}), 400

    with get_db() as db:
        cur = db.execute(
            "INSERT INTO posts (title, content, user_id) VALUES (?, ?, ?)",
            (title, content, session["user_id"]),
        )
        post_id = cur.lastrowid
        post = db.execute("""
            SELECT p.id, p.title, p.content, p.user_id, p.created_at, u.email as author
            FROM posts p JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
        """, (post_id,)).fetchone()

    p = dict(post)
    p["content_html"] = render_markdown(p["content"])
    return jsonify(p), 201


@app.route("/api/posts/<int:post_id>", methods=["GET"])
def api_get_post(post_id):
    with get_db() as db:
        post = db.execute("""
            SELECT p.id, p.title, p.content, p.user_id, p.created_at, u.email as author
            FROM posts p JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
        """, (post_id,)).fetchone()
    if not post:
        return jsonify({"error": "Post not found"}), 404
    p = dict(post)
    p["content_html"] = render_markdown(p["content"])
    return jsonify(p), 200


# ── API: Comments ────────────────────────────────────────────────────────────

@app.route("/api/posts/<int:post_id>/comments", methods=["GET"])
def api_list_comments(post_id):
    with get_db() as db:
        # Verify post exists
        post = db.execute("SELECT id FROM posts WHERE id = ?", (post_id,)).fetchone()
        if not post:
            return jsonify({"error": "Post not found"}), 404
        rows = db.execute("""
            SELECT c.id, c.content, c.user_id, c.post_id, c.created_at, u.email as author
            FROM comments c JOIN users u ON c.user_id = u.id
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC
        """, (post_id,)).fetchall()
    comments = [dict(r) for r in rows]
    return jsonify(comments), 200


@app.route("/api/posts/<int:post_id>/comments", methods=["POST"])
@require_auth
def api_add_comment(post_id):
    with get_db() as db:
        post = db.execute("SELECT id FROM posts WHERE id = ?", (post_id,)).fetchone()
        if not post:
            return jsonify({"error": "Post not found"}), 404

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "Content is required"}), 400

    with get_db() as db:
        cur = db.execute(
            "INSERT INTO comments (content, user_id, post_id) VALUES (?, ?, ?)",
            (content, session["user_id"], post_id),
        )
        comment = db.execute("""
            SELECT c.id, c.content, c.user_id, c.post_id, c.created_at, u.email as author
            FROM comments c JOIN users u ON c.user_id = u.id
            WHERE c.id = ?
        """, (cur.lastrowid,)).fetchone()

    return jsonify(dict(comment)), 201


# ── Session check helper ─────────────────────────────────────────────────────

@app.route("/api/me", methods=["GET"])
def api_me():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 200
    return jsonify({
        "authenticated": True,
        "user_id": session["user_id"],
        "email": session["email"],
    }), 200


# ── Start (dev only) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
