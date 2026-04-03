import os
import time
import psutil
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

# Import the engine from the current root directory
try:
    from tisd_engine_mlx import TISDEngine
except ImportError:
    # Fallback for notebook-style structure if needed
    from notebooks.tisd_engine_mlx import TISDEngine

app = FastAPI(title="Tara AI | TISD")

# Initialize engine (don't load yet to keep startup fast)
engine = TISDEngine(verbose=False)

@app.on_event("startup")
async def startup_event():
    print("🚀 M4 Apple Silicon is warming up...")
    engine.load()
    print("✅ Tara is online and ready.")

@app.get("/ask")
async def ask_tara(q: str, grade: int = 4):
    t0 = time.time()
    
    # NEW: Now returns answer AND sources
    answer, sources = engine.ask(q, grade=grade)
    
    latency = round(time.time() - t0, 2)
    ram = round(psutil.virtual_memory().used / 1e9, 2)
    
    return {
        "answer": answer,
        "sources": sources,
        "telemetry": {
            "latency": f"{latency}s",
            "ram": f"{ram}GB"
        }
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tara | TISD</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            :root { 
                --cartier-red: #8B0000; 
                --cartier-gold: #C5A059; 
                --ivory: #FDFCF9; 
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--ivory); 
                color: #1a1a1a;
                overflow: hidden;
            }
            .heading { font-family: 'Playfair Display', serif; }
            
            /* Chat Bubble Styling */
            .tara-bubble { 
                background: white; 
                border-left: 4px solid var(--cartier-red); 
                box-shadow: 0 2px 10px rgba(0,0,0,0.02);
            }
            .user-bubble { 
                background: #F4F1EA; 
                border-right: 4px solid var(--cartier-gold); 
                text-align: right;
            }
            
            /* Buttons and Inputs */
            .cartier-btn { 
                background: linear-gradient(135deg, #8B0000 0%, #580000 100%); 
                transition: transform 0.2s ease;
            }
            .cartier-btn:active { transform: scale(0.95); }
            
            .input-box {
                box-shadow: 0 10px 40px rgba(0,0,0,0.06);
                border: 1px solid rgba(0,0,0,0.05);
            }

            /* Animations */
            .fade-in { animation: fadeIn 0.6s ease forwards; }
            @keyframes fadeIn { 
                from { opacity: 0; transform: translateY(15px); } 
                to { opacity: 1; transform: translateY(0); } 
            }
            
            /* Scrollbar */
            #chat-history::-webkit-scrollbar { width: 4px; }
            #chat-history::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 10px; }
        </style>
    </head>
    <body class="h-screen flex flex-col items-center">

        <!-- Top Navigation / Stats -->
        <nav class="w-full p-6 flex justify-between items-center fixed top-0 z-50">
            <div class="flex items-center gap-4">
                <select id="grade-select" class="bg-transparent text-[10px] font-bold tracking-widest uppercase border-b border-gray-200 focus:outline-none cursor-pointer">
                    <option value="1">Level 01</option>
                    <option value="2">Level 02</option>
                    <option value="3">Level 03</option>
                    <option value="4" selected>Level 04</option>
                </select>
            </div>
            <div id="stats" class="bg-white/70 backdrop-blur-md px-4 py-2 rounded-full border border-gray-100 flex gap-6 text-[10px] font-mono tracking-tighter text-gray-400 uppercase">
                <span>Silicon: M4 AIR</span>
                <span id="ram-stat">RAM: --</span>
                <span id="lat-stat">Latency: --</span>
            </div>
        </nav>

        <!-- Main Workspace -->
        <main class="w-full max-w-3xl flex-1 flex flex-col relative pt-20">
            
            <!-- Default Landing View -->
            <div id="landing-ui" class="flex-1 flex flex-col items-center justify-center text-center">
                <h1 class="heading text-6xl text-gray-800 mb-4 fade-in">Hello, I'm Tara.</h1>
                <p class="text-gray-400 uppercase tracking-[0.3em] text-[10px] fade-in" style="animation-delay: 0.2s">The Intelligent Student Desk</p>
            </div>

            <!-- Chat History Area (Initially Hidden) -->
            <div id="chat-history" class="hidden flex-1 overflow-y-auto px-4 space-y-8 pb-48 pt-4">
                <!-- Messages dynamic injections -->
            </div>

            <!-- Global Floating Input -->
            <div class="fixed bottom-10 w-full max-w-3xl px-6">
                <div class="input-box bg-white rounded-2xl p-2 flex flex-col relative">
                    <textarea 
                        id="user-input" 
                        rows="1" 
                        placeholder="Ask Tara a question..." 
                        class="w-full p-4 pr-32 focus:outline-none text-lg resize-none bg-transparent"
                    ></textarea>
                    
                    <div class="absolute right-4 bottom-4 flex items-center gap-4">
                        <span class="text-[9px] font-mono text-gray-300 hidden md:block">CTRL + ENTER</span>
                        <button onclick="handleSubmit()" class="cartier-btn text-white p-3 rounded-xl shadow-lg shadow-red-900/20">
                            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </main>

        <script>
            const input = document.getElementById('user-input');
            const chat = document.getElementById('chat-history');
            const landing = document.getElementById('landing-ui');
            const ramStat = document.getElementById('ram-stat');
            const latStat = document.getElementById('lat-stat');

            // Keyboard Shortcut: Ctrl + Enter
            input.addEventListener('keydown', e => {
                if(e.key === 'Enter' && e.ctrlKey) {
                    e.preventDefault();
                    handleSubmit();
                }
            });

            // Auto-expanding Textarea
            input.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });

            async function handleSubmit() {
                const query = input.value.trim();
                const grade = document.getElementById('grade-select').value;
                if(!query) return;

                // 1. UI Transition
                if(!landing.classList.contains('hidden')) {
                    landing.classList.add('hidden');
                    chat.classList.remove('hidden');
                }

                // 2. Add User Bubble
                addBubble(query, 'user');
                
                // 3. Reset Input
                input.value = '';
                input.style.height = 'auto';

                // 4. Add Tara Thinking Bubble
                const taraMsgId = addBubble("Processing...", 'tara');

                try {
                    // 5. Fetch from FastAPI
                    const response = await fetch(`/ask?q=${encodeURIComponent(query)}&grade=${grade}`);
                    const data = await response.json();

                    // 6. Update UI with Answer & Telemetry
                    document.getElementById(taraMsgId).innerText = data.answer;
                    ramStat.innerText = `RAM: ${data.telemetry.ram}`;
                    latStat.innerText = `Latency: ${data.telemetry.latency}`;
                    
                    // Auto-scroll
                    chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });

                } catch (err) {
                    document.getElementById(taraMsgId).innerText = "I'm sorry, I hit a snag. Please check my connection.";
                }
            }

            function addBubble(text, sender) {
                const id = 'msg-' + Date.now();
                const wrapper = document.createElement('div');
                wrapper.className = `flex w-full ${sender === 'user' ? 'justify-end' : 'justify-start'} fade-in`;
                
                const bubble = document.createElement('div');
                bubble.id = id;
                bubble.className = `max-w-[85%] p-5 text-base leading-relaxed ${sender === 'user' ? 'user-bubble' : 'tara-bubble'}`;
                bubble.innerText = text;

                wrapper.appendChild(bubble);
                chat.appendChild(wrapper);
                
                chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
                return id;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)