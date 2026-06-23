# Informe LaTeX

Esta carpeta contiene el informe del Obligatorio 2 de Sistemas Multiagente.

## Estructura

- `main.tex`: documento principal.
- `portada.tex`: portada del informe.
- `Libros.bib`: bibliografia.
- `figures/`: imagenes usadas por el documento.
- `generar_pdf.sh`: script para compilar el PDF.

## Generar el PDF

Desde esta carpeta:

```bash
./generar_pdf.sh
```

Tambien se puede compilar directamente con:

```bash
latexmk -pdf main.tex
```

El resultado se genera como:

```bash
main.pdf
```

## Limpiar archivos auxiliares

```bash
latexmk -C
```

## Requisitos

En macOS, instalar MacTeX:

https://www.tug.org/mactex/

Luego verificar que `latexmk` este disponible:

```bash
latexmk -v
```

