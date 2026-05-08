"""Página 3 — Ranking de árboles con marcado de seleccionados."""
import streamlit as st
import pandas as pd


def render():
    st.header("🏆 Ranking de Árboles")

    results = st.session_state.get("fl_results")
    cfg = st.session_state.get("fl_config")
    
    if not results:
        st.warning("⚠️ Ejecuta primero el experimento en la página '▶️ Ejecutar'.")
        return
    
    # Get dynamic threshold from config
    conv_thr = cfg.get("aggregation", {}).get("convergence", 0.002) if cfg else 0.002

    sid = results.strategy_id.lower()
    is_per_client = sid.startswith("s5") or sid.startswith("s6") or sid.startswith("s7")
    n_total   = len(results.all_tree_entries)
    # Ensure client_id is string when looking up selected_ids
    n_sel     = sum(1 for e in results.all_tree_entries
                    if e.tree_local_id in results.selected_ids.get(str(e.client_id), []))

    info_text = (
        f"**Estrategia:** `{sid}` &nbsp;|&nbsp; "
        f"**Total árboles candidatos:** {n_total} &nbsp;|&nbsp; "
        f"**Seleccionados para bosque global:** {n_sel} &nbsp;|&nbsp; "
        f"**Descartados por Progressive:** {n_total - n_sel}"
    )
    # Show detailed convergence info if available (S2-S7, PW)
    if sid not in ["s1_simple_pool", "s1"] and hasattr(results, 'round_logs') and results.round_logs:
        st.info(info_text)
        logs = results.round_logs
        
        if results.convergence_round is not None:
            # It converged!
            last_idx = results.convergence_round - 1
            if 0 < last_idx < len(logs):
                current_acc = logs[last_idx].get('accuracy', 0.0)
                prev_acc = logs[last_idx - 1].get('accuracy', 0.0)
                improvement = current_acc - prev_acc
                
                st.success(f"🎯 **Convergencia alcanzada en el episodio {results.convergence_round}**.\n\n"
                           f"La mejora final fue de `{improvement:.5f}` (Umbral: `{conv_thr}`).")
            else:
                st.success(f"🎯 **Parada Temprana**: El modelo global convergió en el episodio {results.convergence_round}.")
        else:
            # Did not converge
            if len(logs) >= 2:
                current_acc = logs[-1].get('accuracy', 0.0)
                prev_acc = logs[-2].get('accuracy', 0.0)
                improvement = current_acc - prev_acc
                
                st.warning(f"⚠️ **Selección completa ({n_sel} árboles)**.\n\n"
                           f"No se alcanzó la convergencia. Variación última: `{improvement:.5f}` (Umbral: `{conv_thr}`).")
            else:
                st.warning(f"⚠️ **Sin Convergencia**: Se seleccionaron todos los árboles candidatos ({n_sel}). No hubo suficientes episodios para evaluar la mejora continua.")
    else:
        st.info(info_text)

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
        sel = entry.tree_local_id in results.selected_ids.get(str(entry.client_id), [])
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
    if sid == "s1_simple_pool" or sid == "s1":
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

    # ── Detalle por cliente (Scalable View) ───────────────────────────────────
    st.subheader("🔍 Detalle por Cliente")
    sel_clients = sorted(df["Cliente"].unique())
    
    if not sel_clients:
        st.info("No hay datos de clientes disponibles.")
    else:
        # Client selector for scalability
        selected_cid = st.selectbox("Seleccionar cliente para ver sus árboles", 
                                   ["Todos"] + [str(c) for c in sel_clients],
                                   index=0)
        
        if selected_cid == "Todos":
            st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True)
        else:
            sub = df[df["Cliente"].astype(str) == selected_cid]
            n_s = sub["_sel"].sum()
            st.write(f"Mostrando **{len(sub)}** árboles del cliente **{selected_cid}** ({int(n_s)} seleccionados para el bosque global).")
            st.dataframe(sub[display_cols].reset_index(drop=True), use_container_width=True)
