import os
from typing import TypedDict
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from mapel import MAPEL_PER_JURUSAN
from kompetensi import KOMPETENSI_PER_JURUSAN

# ==========================================
# 1. DEFINISI STATE (Shared Memory)
# ==========================================
class CareerCoachState(TypedDict):
    nama_siswa: str
    jurusan: str
    pekerjaan_impian: str
    teks_nilai: str
    teks_kompetensi: str
    hasil_agent_1: str
    hasil_agent_2: str
    hasil_agent_3: str
    laporan_final: str


# ==========================================
# 2. DEFINISI NODE / AGENT
# ==========================================
MODEL_NAME = "llama-3.3-70b-versatile"

def get_llm():
    return ChatGroq(
        model_name=MODEL_NAME,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2
    )

def agent_analis_rapor(state: CareerCoachState) -> dict:
    llm = get_llm()
    prompt = f"""Anda adalah Guru Produktif Senior SMK untuk Jurusan {state['jurusan']}.
    Nama Siswa: {state['nama_siswa']}
    Transkrip Nilai:
    {state['teks_nilai']}
    Standar Kompetensi Kejuruan & PKL:
    {state['teks_kompetensi']}

    Tugas Anda:
    1. Analisis seluruh nilai mata pelajaran.
    2. Khusus mata pelajaran produktif dan PKL, kaitkan capaian nilai dengan standar kompetensi.
    3. Jika nilai >= 80, uraikan kompetensi yang dikuasai. Jika kurang, sebutkan yang perlu ditingkatkan.
    4. Berikan laporan ringkas dalam format Markdown."""
    
    response = llm.invoke(prompt)
    return {"hasil_agent_1": response.content}


def agent_hrd_matcher(state: CareerCoachState) -> dict:
    llm = get_llm()
    prompt = f"""Anda adalah HRD Profesional untuk posisi '{state['pekerjaan_impian']}' dan jurusan '{state['jurusan']}'.
    Profil Kompetensi Siswa:
    {state['hasil_agent_1']}

    Tugas Anda:
    1. Jabarkan DAFTAR EKSPEKTASI UMUM INDUSTRI untuk posisi {state['pekerjaan_impian']}.
    2. Berikan perkiraan PERSENTASE KECOCOKAN (Match Rate dalam %).
    3. Rincikan daftar GAP SKILL siswa secara detail.
    4. Berikan DISCLAIMER bahwa standar rekrutmen dapat bervariasi di tiap perusahaan."""
    
    response = llm.invoke(prompt)
    return {"hasil_agent_2": response.content}


def agent_mentor_belajar(state: CareerCoachState) -> dict:
    llm = get_llm()
    prompt = f"""Anda adalah Mentor Profesional / Desainer Pembelajaran Digital.
    Jurusan: '{state['jurusan']}', Posisi Target: '{state['pekerjaan_impian']}'
    Daftar GAP SKILL Siswa:
    {state['hasil_agent_2']}

    Tugas Anda:
    Susun rekomendasi topik pembelajaran spesifik dan sertakan kata kunci (keyword) pencarian video tutorial YouTube atau kursus online untuk menambal kekurangan tersebut."""
    
    response = llm.invoke(prompt)
    return {"hasil_agent_3": response.content}


def agent_guru_bk(state: CareerCoachState) -> dict:
    llm = get_llm()
    prompt = f"""Anda adalah Guru Bimbingan Konseling (BK) di SMKN 1 Kasreman.
    Rangkum hasil evaluasi berikut menjadi Laporan Konseling Karier Akhir untuk {state['nama_siswa']} ({state['jurusan']}) target '{state['pekerjaan_impian']}':

    - Nilai Rapor: {state['teks_nilai']}
    - Evaluasi Guru Produktif: {state['hasil_agent_1']}
    - Analisis HRD & Gap Skill: {state['hasil_agent_2']}
    - Rekomendasi Belajar: {state['hasil_agent_3']}

    Struktur Laporan:
    1. PENGANTAR
    2. POTRET KOMPETENSI SISWA (sertakan ringkasan nilai dan capaian unit kompetensi)
    3. ANALISIS KESIAPAN DUNIA KERJA (Match Rate, Gap Skill, & Catatan industri)
    4. RENCANA AKSI MANDIRI
    5. KATA-KATA MOTIVASI PENUTUP"""
    
    response = llm.invoke(prompt)
    return {"laporan_final": response.content}


# ==========================================
# 3. PENYUSUNAN GRAPH WORKFLOW
# ==========================================
def build_career_graph():
    workflow = StateGraph(CareerCoachState)

    # Daftarkan Nodes
    workflow.add_node("analis_rapor", agent_analis_rapor)
    workflow.add_node("hrd_matcher", agent_hrd_matcher)
    workflow.add_node("mentor_belajar", agent_mentor_belajar)
    workflow.add_node("guru_bk", agent_guru_bk)

    # Rangkai Edges (Alur Kerja Sekuensial)
    workflow.add_edge(START, "analis_rapor")
    workflow.add_edge("analis_rapor", "hrd_matcher")
    workflow.add_edge("hrd_matcher", "mentor_belajar")
    workflow.add_edge("mentor_belajar", "guru_bk")
    workflow.add_edge("guru_bk", END)

    return workflow.compile()


# ==========================================
# 4. STREAMLIT UI & EKSEKUSI GRAPH
# ==========================================
st.set_page_config(page_title="Smart Career Path SMKN 1 Kasreman", page_icon="🎓", layout="centered")

st.title("🎓 Smart Career Path (LangGraph Edition)")
st.subheader("Asisten Konseling Karier Virtual SMKN 1 Kasreman")
st.markdown("---")

if "GROQ_API_KEY" in os.environ:
    st.sidebar.success("🔑 Groq API Key terdeteksi.")
else:
    api_key_input = st.sidebar.text_input("Masukkan Groq API Key:", type="password")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

st.header("📋 Input Data Siswa")
nama_siswa = st.text_input("Nama Lengkap Siswa:", placeholder="Contoh: Muhammad Reyhan")
jurusan_pilihan = st.selectbox("Pilih Jurusan:", options=list(MAPEL_PER_JURUSAN.keys()))
pekerjaan_impian = st.text_input("Pekerjaan Impian:", placeholder="Contoh: Junior Backend Developer")

current_mapel_list = MAPEL_PER_JURUSAN[jurusan_pilihan]
col1, col2 = st.columns(2)
dict_nilai_input = {}

for idx, (label_formal, key) in enumerate(current_mapel_list):
    target_col = col1 if idx % 2 == 0 else col2
    with target_col:
        val = st.number_input(
            label=f"{label_formal}:",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1,
            format="%.2f",
            key=f"{jurusan_pilihan}_{key}"
        )
        dict_nilai_input[label_formal] = val

if st.button("🚀 Mulai Analisis Karier Saya", type="primary"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("API Key Groq belum diatur.")
    elif not nama_siswa or not pekerjaan_impian:
        st.warning("Mohon isi Nama Lengkap Siswa dan Pekerjaan Impian.")
    else:
        with st.spinner("Graph Multi-Agent sedang mengeksekusi pipeline..."):
            try:
                # Siapkan Data Input
                teks_nilai = "\n".join([f"- {k}: {v:.2f}" for k, v in dict_nilai_input.items()])
                kompetensi_jurusan = KOMPETENSI_PER_JURUSAN[jurusan_pilihan]
                teks_kompetensi = ""
                for mapel_name, daftar_poin in kompetensi_jurusan.items():
                    teks_kompetensi += f"\n📌 {mapel_name}:\n" + "\n".join([f"  * {p}" for p in daftar_poin])

                initial_state = {
                    "nama_siswa": nama_siswa,
                    "jurusan": jurusan_pilihan,
                    "pekerjaan_impian": pekerjaan_impian,
                    "teks_nilai": teks_nilai,
                    "teks_kompetensi": teks_kompetensi,
                    "hasil_agent_1": "",
                    "hasil_agent_2": "",
                    "hasil_agent_3": "",
                    "laporan_final": ""
                }

                # Inisialisasi dan Jalankan Graph
                app = build_career_graph()
                final_output = app.invoke(initial_state)

                st.success("Analisis Selesai!")
                st.markdown("---")
                st.header("📊 Hasil Analisis Konseling Karier")
                st.markdown(final_output["laporan_final"])

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
