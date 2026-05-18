# Образ фронтенда SpectrumAI (Этап 15, глава 4.6).
# Multi-stage: builder собирает SPA через Vite, runtime — nginx alpine
# раздаёт статику + проксирует /api/* на backend:8000.

FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# VITE_API_URL="" → axios baseURL пустой → все запросы относительные
# (/api/v1/...), проксируются nginx на backend в той же docker-сети.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build


# -----------------------------------------------------------------------------
FROM nginx:alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
