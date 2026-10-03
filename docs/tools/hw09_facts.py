"""Measure the numbers behind docs/HW09/FACTS.md.

Run from HW09/ (needs the GPU, checkpoint.pth, food/ and the HF models):
    ../.venv/bin/python ../docs/tools/hw09_facts.py [section ...]
Sections: env model predict lime saliency smooth filter ig bert embed (default: all)
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, '.')
from model import Classifier
from dataset import FoodDataset, get_paths_labels

CATEGORIES = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat',
              'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']


def header(name):
    print(f'\n===== {name} =====')


def load():
    model = Classifier().cuda()
    model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
    model.eval()
    paths, labels = get_paths_labels('./food/')
    images, labels = FoodDataset(paths, labels, mode='eval').getbatch(range(10))
    return model, paths, images, labels


def env():
    header('env')
    import lime, skimage, sklearn, transformers, matplotlib, torchvision
    for m in [torch, torchvision, np, matplotlib, lime, skimage, sklearn, transformers]:
        print(m.__name__, getattr(m, '__version__', '?'))
    print('gpu', torch.cuda.get_device_name(0))


def model_facts(model, images):
    header('model')
    for i, layer in enumerate(model.cnn):
        print(f'cnn[{i}]', layer)
    print('fc', model.fc)
    n = sum(p.numel() for p in model.parameters())
    print('params', n, 'cnn', sum(p.numel() for p in model.cnn.parameters()), 'fc', sum(p.numel() for p in model.fc.parameters()))
    x = images.cuda()
    with torch.no_grad():
        for i, layer in enumerate(model.cnn):
            x = layer(x)
            if isinstance(layer, torch.nn.MaxPool2d) or i in (6, 23):
                print(f'after cnn[{i}] {type(layer).__name__}', tuple(x.shape))


def predict_facts(model, paths, images, labels):
    header('predict')
    print('image tensor', tuple(images.shape), images.dtype, 'min', images.min().item(), 'max', images.max().item())
    with torch.no_grad():
        logits = model(images.cuda()).cpu()
    prob = logits.softmax(1)
    correct = 0
    for i in range(10):
        top = prob[i].topk(3)
        pred = top.indices[0].item()
        correct += pred == labels[i].item()
        tops = ', '.join(f'{CATEGORIES[c]}({c}) {p:.4f}' for p, c in zip(top.values.tolist(), top.indices.tolist()))
        print(f'{i} {paths[i]} label {labels[i].item()} {CATEGORIES[labels[i]]} | pred {pred} | p(label) {prob[i, labels[i]]:.4f} | top3 {tops}')
    print('correct', correct, '/ 10')


def lime_facts(model, images, labels):
    header('lime')
    from skimage.segmentation import slic
    from lime import lime_image
    calls = []

    def predict(input):
        calls.append(input.shape)
        input = torch.FloatTensor(input).permute(0, 3, 1, 2)
        with torch.no_grad():
            return model(input.cuda()).cpu().numpy()

    def segmentation(input):
        return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)

    np.random.seed(16)
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        calls.clear()
        t = time.time()
        exp = lime_image.LimeImageExplainer().explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        dt = time.time() - t
        segs = exp.segments
        w = exp.local_exp[label.item()]
        big = [(s, v) for s, v in w if abs(v) >= 0.05]
        shown = big[:11]
        print(f'img {idx}: segments {len(np.unique(segs))} (ids {segs.min()}..{segs.max()}), predict calls {len(calls)} first batch {calls[0]} last {calls[-1]}, '
              f'time {dt:.2f}s, top_labels {exp.top_labels}, score {exp.score[label.item()] if isinstance(exp.score, dict) else exp.score:.4f}')
        print(f'   |w|>=0.05: {len(big)} (pos {sum(v > 0 for _, v in big)}, neg {sum(v < 0 for _, v in big)}); shown(num_features=11): pos {sum(v > 0 for _, v in shown)} neg {sum(v < 0 for _, v in shown)}')
        print('   top5', [(int(s), round(float(v), 4)) for s, v in w[:5]])


def saliency_facts(model, images, labels):
    header('saliency')
    x = images.cuda().requires_grad_()
    loss = torch.nn.CrossEntropyLoss()(model(x), labels.cuda())
    loss.backward()
    print('mean CE loss over 10 images', round(loss.item(), 4))
    g = x.grad.abs().cpu()
    sal, which = g.max(dim=1)
    print('grad shape', tuple(x.grad.shape), '-> saliency', tuple(sal.shape))
    for i in range(10):
        counts = [(which[i] == c).float().mean().item() for c in range(3)]
        print(f'img {i}: raw max {sal[i].max():.3e} mean {sal[i].mean():.3e}  argmax channel R/G/B share {counts[0]:.2f}/{counts[1]:.2f}/{counts[2]:.2f}')
    m = sal.flatten(1).max(1).values
    print('ratio of largest to smallest per-image max', round((m.max() / m.min()).item(), 1))


def smooth_facts(model, images, labels):
    header('smooth')
    torch.manual_seed(0)
    for i in range(10):
        x = images[i]
        rng = (x.max() - x.min()).item()
        sigma = 0.4 / rng
        print(f'img {i}: max-min {rng:.4f} sigma {sigma:.4f} noise std used (sigma**2) {sigma ** 2:.4f}  paper-style 0.4*(max-min) {0.4 * rng:.4f}')
    x, y = images[0], labels[0]
    for epoch in (1, 10, 50, 500):
        smooth = np.zeros((1, 3, 128, 128))
        t = time.time()
        for _ in range(epoch):
            noise = x.new_empty(x.size()).normal_(0, (0.4 / (x.max() - x.min()).item()) ** 2)
            x_mod = (x + noise).unsqueeze(0).cuda().requires_grad_()
            torch.nn.CrossEntropyLoss()(model(x_mod), y.cuda().unsqueeze(0)).backward()
            smooth += x_mod.grad.abs().cpu().numpy()
        avg = smooth / epoch
        print(f'img 0 epoch {epoch}: time {time.time() - t:.2f}s, before normalize min {avg.min():.3e} max {avg.max():.3e}')


def filter_facts(model, images):
    header('filter')
    from torch.optim import Adam
    store = {}
    for cnnid in (6, 23):
        h = model.cnn[cnnid].register_forward_hook(lambda m, i, o: store.__setitem__('a', o))
        with torch.no_grad():
            model(images.cuda())
        a = store['a'][:, 0]
        print(f'cnn[{cnnid}] = {model.cnn[cnnid]}; activation {tuple(store["a"].shape)} filter0 map {tuple(a.shape)}')
        print('   filter0 sum per image', [round(v, 1) for v in a.flatten(1).sum(1).tolist()], 'zero fraction', round((a == 0).float().mean().item(), 3))
        x = images.cuda().requires_grad_()
        opt = Adam([x], lr=0.1)
        log = []
        for it in range(100):
            opt.zero_grad()
            model(x)
            obj = -store['a'][:, 0].sum()
            obj.backward()
            opt.step()
            if it in (0, 1, 9, 49, 99):
                log.append((it, round(-obj.item(), 1)))
        print('   activation sum (all 10) at iter', log)
        print(f'   optimized x range {x.min().item():.2f} .. {x.max().item():.2f}')
        h.remove()


def ig_facts(model, images, labels):
    header('ig')
    print('alphas (step/steps, steps=10):', [s / 10 for s in range(10)])
    for i in range(10):
        x = images[i:i + 1].cuda()
        t = labels[i].item()
        grads = []
        for s in range(10):
            xb = (x * s / 10).detach().requires_grad_()
            model(xb)[0, t].backward()
            grads.append(xb.grad)
        avg = torch.stack(grads).mean(0)
        with torch.no_grad():
            fx = model(x)[0, t].item()
            f0 = model(torch.zeros_like(x))[0, t].item()
        print(f'img {i}: logit f(x) {fx:.3f} f(0) {f0:.3f} diff {fx - f0:.3f} | sum(avg_grad) {avg.sum().item():.3f} sum(avg_grad*x) {(avg * x).sum().item():.3f}')
    # same check with more steps and the midpoint rule
    x = images[0:1].cuda()
    t = labels[0].item()
    for steps in (10, 50, 200):
        acc = torch.zeros_like(x)
        for s in range(steps):
            xb = (x * (s + 0.5) / steps).detach().requires_grad_()
            model(xb)[0, t].backward()
            acc += xb.grad / steps
        print(f'img 0 midpoint steps {steps}: sum(avg_grad*x) {(acc * x).sum().item():.3f}')


def bert_facts():
    header('bert')
    sys.argv = sys.argv[:1]
    import bert_hidden_states as B
    from transformers import BertForQuestionAnswering, BertTokenizerFast
    from sklearn.decomposition import PCA
    tok = BertTokenizerFast.from_pretrained(B.qa_model_name)
    model = BertForQuestionAnswering.from_pretrained(B.qa_model_name, output_hidden_states=True).eval()
    print('model', B.qa_model_name, 'vocab', tok.vocab_size, 'do_lower_case', getattr(tok, 'do_lower_case', '?'))
    for q in (1, 2, 3):
        inputs = tok(B.questions[q - 1], B.contexts[q - 1], return_tensors='pt')
        ids = inputs['input_ids'][0].tolist()
        qe = ids.index(102) - 1
        with torch.no_grad():
            out = model(**inputs)
        hs = out.hidden_states
        words = [tok.decode(t) for t in ids]
        ans = [i for i, w in enumerate(words) if w in B.answers[q - 1].split()]
        s, e = out.start_logits[0].argmax().item(), out.end_logits[0].argmax().item()
        print(f'Q{q}: seq len {len(ids)}, question 1..{qe}, context {qe + 2}..{len(ids) - 2}, hidden_states {len(hs)} x {tuple(hs[0].shape)}')
        print(f'   answer-matched token positions {ans} -> {[words[i] for i in ans]}; model predicts span {s}..{e} = {tok.decode(ids[s:e + 1])!r}')
        print('   tokens', words)
        print('   PCA explained variance (2 comps) per layer 1..12', [round(float(PCA(n_components=2, random_state=0).fit(h[0]).explained_variance_ratio_.sum()), 2) for h in hs[1:]])


def embed_facts():
    header('embed')
    import bert_embedding as E
    from transformers import BertModel, BertTokenizerFast
    model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True).eval()
    tok = BertTokenizerFast.from_pretrained('bert-base-chinese')
    toks = [tok(s, return_tensors='pt') for s in E.sentences]
    with torch.no_grad():
        outs = [model(**t) for t in toks]
    idx2 = [5, 3, 1, 9, 3, 1, 1, 5, 1, 1]
    for i, s in enumerate(E.sentences):
        t = toks[i]
        pieces = tok.convert_ids_to_tokens(t['input_ids'][0])
        a = t.word_to_tokens(E.select_word_index[i]).start
        b = t.word_to_tokens(idx2[i]).start
        print(f'{i} {s} | {len(pieces)} tokens | index {E.select_word_index[i]} -> token {a} {pieces[a]!r} | index {idx2[i]} -> token {b} {pieces[b]!r}')
        print('   tokens', pieces)
    fruit, company = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    print('groups: fruit', fruit, 'company', company)
    for metric in (E.euclidean_distance, E.cosine_similarity):
        for idx_name, idx in (('蘋', E.select_word_index), ('果', idx2)):
            for layer in range(13):
                emb = np.stack([outs[i].hidden_states[layer][0][toks[i].word_to_tokens(idx[i]).start].numpy() for i in range(10)])
                from sklearn.metrics import pairwise_distances
                m = pairwise_distances(emb, metric=metric)

                def mean(g1, g2):
                    v = [m[a, b] for a in g1 for b in g2 if a != b]
                    return sum(v) / len(v)
                line = f'{metric.__name__} {idx_name} layer {layer:2d}: within-fruit {mean(fruit, fruit):.3f} within-company {mean(company, company):.3f} across {mean(fruit, company):.3f}'
                print(line)
                if layer == 12 and idx_name == '蘋':
                    print('   matrix\n' + np.array2string(m, precision=2, max_line_width=200))


if __name__ == '__main__':
    want = sys.argv[1:] or ['env', 'model', 'predict', 'lime', 'saliency', 'smooth', 'filter', 'ig', 'bert', 'embed']
    if 'env' in want:
        env()
    if set(want) & {'model', 'predict', 'lime', 'saliency', 'smooth', 'filter', 'ig'}:
        model, paths, images, labels = load()
        if 'model' in want: model_facts(model, images)
        if 'predict' in want: predict_facts(model, paths, images, labels)
        if 'lime' in want: lime_facts(model, images, labels)
        if 'saliency' in want: saliency_facts(model, images, labels)
        if 'smooth' in want: smooth_facts(model, images, labels)
        if 'filter' in want: filter_facts(model, images)
        if 'ig' in want: ig_facts(model, images, labels)
    if 'bert' in want: bert_facts()
    if 'embed' in want: embed_facts()
