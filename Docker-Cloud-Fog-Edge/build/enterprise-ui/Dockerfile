FROM node:20-alpine AS builder
WORKDIR /app
COPY enterprise-ui/package.json ./package.json
RUN npm install
COPY enterprise-ui .
RUN npm run build

FROM nginx:1.27-alpine
COPY build/enterprise-ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
