"""Página 3 — Ranking de árboles con marcado de seleccionados."""
import streamlit as st
import pandas as pd


# Paleta de colores para clientes (hasta 20)
_COLORS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7", "#D7BDE2",
    "#FADBD8", "#D5F5E3", "#FCF3CF", "#D6EAF8", "#E8DAEF",
    "#FDEBD0", "#D1F2EB", "#FDFEFE", "#EBF5FB", "#FEF9E7",
    "#F9EBEA", "#EAF2FF", "#E8F8F5", "#FDF2E9", "#F4ECF7",
]


def render():
    st.header("🏆 Ranking de Árboles")

    results = st.session_state.get("fl_results")
    if not results:
        st.warning("⚠️ Ejecuta primero el experimento en la página '▶️ Ejecutar'.")
        return

    sid = results.strategy_id
    is_per_client = sid.startswith("s5") or sid.startswith("s6") or sid.startswith("s7")
    n_total   = len(results.all_tree_entries)
    n_sel     = sum(1 for e in results.all_tree_entries
                    if e.tree_local_id in results.selected_ids.get(e.client_id, []))

    st.info(
        f"**Estrategia:** `{sid}` &nbsp;|&nbsp; "
        f"**Total árboles candidatos:** {n_total} &nbsp;|&nbsp; "
        f"**Seleccionados para bosque global:** {n_sel} &nbsp;|&nbsp; "
        f"**Descartados por Progressive:** {n_total - n_sel}"
    )

    # ── Leyenda ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    c1.success("🟢  Árbol seleccionado — incluido en el bosque global")
    c2.error("🔴  Árbol descartado — no superó el criterio de parada")
    st.caption("El ID del árbol tiene formato `{client_id}_tree{local_id}`. "
               "Este `local_id` es el que usa el No-Repeat Merge en la actualización del cliente.")

    st.divider()

    # ── Construir tabla ───────────────────────────────────────────────────────
    rows = []
    for rank, entry in enumerate(results.all_tree_entries, 1):
        sel = entry.tree_local_id in results.selected_ids.get(entry.client_id, [])
        rows.append({
            "Rank":       rank,
            "ID árbol":   f"{entry.client_id}_tree{entry.tree_local_id}",
            "Cliente":    entry.client_id,
            "Accuracy":   round(entry.accuracy, 4),
            "Macro-F1":   round(entry.macro_f1, 4),
            "PCD":        round(entry.pcd, 4),
            "Score":      round(getattr(entry, 'score', entry.accuracy), 4),
            "Estado":     "✅ Seleccionado" if sel else "❌ Descartado",
            "_sel":       sel,
            "_cid":       entry.client_id,
        })

    df = pd.DataFrame(rows)

    # Para estrategia s1, ordenar por cliente
    if sid == "s1_simple_pool":
        df = df.sort_values(["Cliente", "Rank"], ascending=[True, True]).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)  # Reasignar rank después de ordenar

    # ── Mostrar tabla ─────────────────────────────────────────────────────────
    display_cols = ["Rank", "ID árbol", "Cliente", "Accuracy", "Macro-F1", "PCD", "Score", "Estado"]
    df_display = df[display_cols].copy()
    
    st.subheader("Tabla de Árboles")
    st.dataframe(df_display, use_container_width=True, height=500)

    # ── Stats rápidas ─────────────────────────────────────────────────────────
    st.divider()
    n_shown = len(df)
    n_sel_shown = df["_sel"].sum()
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Mostrados", n_shown)
    cB.metric("Seleccionados", int(n_sel_shown))
    cC.metric("Descartados", int(n_shown - n_sel_shown))
    cD.metric("% Selección", f"{100*n_sel_shown/n_shown:.1f}%" if n_shown else "—")

    # ── Por cliente ───────────────────────────────────────────────────────────
    if is_per_client:
        st.subheader("Detalle por cliente")
        for cid in sel_clients:
            sub = df[df["Cliente"] == cid]
            if sub.empty:
                continue
            n_s = sub["_sel"].sum()
            with st.expander(f"👤 {cid} — {len(sub)} árboles, {int(n_s)} seleccionados"):
                st.dataframe(sub[display_cols].reset_index(drop=True),
                             use_container_width=True)
