# 📦 Guía: Crear Repositorio en GitHub

Esta guía te ayuda a crear un repositorio en GitHub y pushear este proyecto.

## Paso 1: Crear Repositorio en GitHub

### Por navegador (Recomendado)

1. Ir a https://github.com/new
2. Llenar formulario:
   - **Repository name:** `federated_proactive_forest`
   - **Description:** "Sistema de Aprendizaje Federado Horizontal con 7 estrategias de agregación basado en Proactive Forest"
   - **Visibility:** Public (o Private si prefieres)
   - ✅ No inicializar con README (ya lo tenemos)
3. Click "Create repository"

### Copiar URL SSH o HTTPS

En la página del repo, verás:
```
git@github.com:tu_usuario/federated_proactive_forest.git  (SSH)
# o
https://github.com/tu_usuario/federated_proactive_forest.git  (HTTPS)
```

---

## Paso 2: Inicializar Git Localmente

### En la carpeta del proyecto

```bash
cd c:\Users\Adrián\ Rodríguez\Videos\federated_proactive_forest
```

### Inicializar repositorio

```bash
# Inicializar git
git init

# Configurar nombre y email (si no lo has hecho antes)
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@ejemplo.com"

# Verificar
git config user.name
git config user.email
```

---

## Paso 3: Agregar Archivos y Hacer Commit Inicial

```bash
# Ver qué archivos se van a agregar
git status

# Agregar TODOS los archivos (respeta .gitignore)
git add .

# Crear commit inicial
git commit -m "Initial commit: Federated Proactive Forest with 7 aggregation strategies"

# Verificar
git log --oneline
```

**Salida esperada:**
```
abc1234 Initial commit: Federated Proactive Forest with 7 aggregation strategies
```

---

## Paso 4: Conectar con GitHub y Hacer Push

### Agregar remoto

```bash
# Reemplaza TU_USUARIO con tu nombre de usuario de GitHub
git remote add origin https://github.com/TU_USUARIO/federated_proactive_forest.git

# Verificar
git remote -v
```

**Salida esperada:**
```
origin  https://github.com/TU_USUARIO/federated_proactive_forest.git (fetch)
origin  https://github.com/TU_USUARIO/federated_proactive_forest.git (push)
```

### Cambiar rama a `main` (GitHub por defecto usa `main`, no `master`)

```bash
git branch -M main
```

### Hacer push

```bash
git push -u origin main
```

**Primera vez:** Puede pedir autenticación
- Si usas HTTPS: ingresa tu usuario + token de GitHub
- Si usas SSH: asegúrate de tener clave SSH configurada

---

## Paso 5: Verificar en GitHub

1. Abre https://github.com/tu_usuario/federated_proactive_forest
2. Deberías ver:
   - ✅ Todos los archivos
   - ✅ README.md renderizado
   - ✅ .gitignore activo
   - ✅ Commits en la rama `main`

---

## Opciones Avanzadas

### Si usas Windows y tienes problemas

**Usar Git Bash en lugar de PowerShell:**

1. Descargar Git para Windows: https://git-scm.com/download/win
2. Instalar (acepta todas las opciones por defecto)
3. Abrir "Git Bash" en la carpeta del proyecto
4. Seguir los pasos anteriores

### Si necesitas autenticación con SSH

```bash
# Generar clave SSH
ssh-keygen -t ed25519 -C "tu_email@ejemplo.com"

# Seguir instrucciones, Enter para todo

# Copiar clave pública
cat ~/.ssh/id_ed25519.pub | clip

# En GitHub:
# Settings → SSH and GPG keys → New SSH key
# Pegar clave
```

Luego usar URL SSH para push.

---

## Flujo Normal Después

Una vez que el repo esté en GitHub:

```bash
# Ver cambios
git status

# Agregar cambios
git add .

# Commit
git commit -m "Descripción breve del cambio"

# Push
git push

# Jalar cambios (si trabajas en otra máquina)
git pull
```

---

## Comandos Útiles

```bash
# Ver historial de commits
git log --oneline

# Ver diferencias
git diff

# Ver ramas
git branch

# Crear rama nueva
git checkout -b feature/mi-caracteristica

# Cambiar de rama
git checkout main

# Ver remoto
git remote -v

# Cambiar URL remoto
git remote set-url origin NEW_URL
```

---

## Troubleshooting

### Error: "fatal: not a git repository"
```bash
# Asegúrate de estar en la carpeta correcta
cd federated_proactive_forest
git init
```

### Error: "Please tell me who you are"
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@ejemplo.com"
```

### Error: "authentication failed"
```bash
# Para HTTPS: Usar token en lugar de password
# Para SSH: Generar y configurar clave

# Ver qué autenticación usas
git remote -v

# Cambiar a HTTPS si tienes problemas
git remote set-url origin https://github.com/usuario/repo.git
```

### El push tarda/se queda colgado
```bash
# Ctrl+C para cancelar
# Luego reintentar
git push
```

---

## Checklist Final

- [ ] Repositorio creado en GitHub
- [ ] Git inicializado localmente (`git init`)
- [ ] Primer commit realizado (`git commit`)
- [ ] Remoto agregado (`git remote add origin ...`)
- [ ] Cambio a rama `main` (`git branch -M main`)
- [ ] Push realizado (`git push -u origin main`)
- [ ] Verificado en GitHub (archivos visibles)
- [ ] README renderizado correctamente
- [ ] .gitignore activo (no hay archivos innecesarios)

---

## Próximos Pasos

1. **Crear Issues** — Para bugs o features
2. **Crear Milestones** — Para planificar releases
3. **Agregar Topics** — `federated-learning`, `machine-learning`, `python`
4. **Documentación** — Wikis, Guías adicionales
5. **CI/CD** — GitHub Actions para testing

---

¡Listo! Tu proyecto está en GitHub 🎉

